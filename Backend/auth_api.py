"""
Backend API for the NSBM Admin AI portal (FastAPI): sign-in and the AI chat.

Start it from the project folder, with the Python 3.11+ venv active:
    python -m uvicorn auth_api:app --reload --port 8001 --app-dir Backend

Sign-in has two steps: email + password, then a 6-digit code sent by email.
The chat (POST /api/chat) sends a signed-in user's question to the team's
AcademicAgent (src/agent) through chat_service.py.

Endpoints
    POST /api/auth/register      {"full_name", "email", "password"}  -> emails a code
    POST /api/auth/login         {"email", "password"}               -> emails a code
    POST /api/auth/resend-code   {"email"}                           -> emails a new code
    POST /api/auth/verify-otp    {"email", "code"}  -> {"token", "user", "is_new_user", ...}
    GET  /api/auth/me            (Bearer token)     -> the signed-in user
    PUT  /api/auth/profile       {"full_name"}      -> changes the name
    POST /api/auth/logout        (Bearer token)     -> 204
    GET  /api/admin/users        (admin token only) -> every account
    POST /api/chat               {"question"}  (Bearer token)
                                 -> {"answer", "answerable", "sources", "trace", ...}
    GET  /api/health                                 -> {"status": "ok", "chat": {...}, ...}

Errors always look like {"detail": "message for the user", "code": "invalid_credentials"}.

Any endpoint can require a signed-in user or an admin:
    def chat(body: ChatIn, user: dict = Depends(get_current_user)): ...
    def reindex(user: dict = Depends(require_admin)): ...   # only admins get here
"""

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Find auth_service.py next to this file, whichever folder uvicorn is started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Header, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from auth_service import AuthError, AuthService, Settings  # noqa: E402
import chat_service  # noqa: E402
from chat_service import ChatError  # noqa: E402

settings = Settings.from_env()   # also reads .env (GEMINI_API_KEY, SMTP_..., ...)
auth = AuthService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the AI agent in the background, so the server starts at once
    # and the first question doesn't wait for the search model.
    threading.Thread(target=chat_service.warm_up, daemon=True).start()
    yield


app = FastAPI(title="NSBM Admin AI - API", lifespan=lifespan)

# The UI is opened from VS Code (Go Live at http://127.0.0.1:5500, or F5 as a file),
# so the API has to accept requests from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# ----------------------------------------------------------------- errors -- #

@app.exception_handler(AuthError)
async def handle_auth_error(request: Request, exc: AuthError) -> JSONResponse:
    body = {"detail": exc.message, "code": exc.code}
    if exc.field:
        body["field"] = exc.field
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(status_code=exc.status, content=body, headers=headers)


@app.exception_handler(ChatError)
async def handle_chat_error(request: Request, exc: ChatError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.message, "code": exc.code})


@app.exception_handler(RequestValidationError)
async def handle_bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": "Some fields are missing or invalid.", "code": "invalid_request"},
    )


# --------------------------------------------------------------- requests -- #

class RegisterIn(BaseModel):
    full_name: str
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class EmailIn(BaseModel):
    email: str


class VerifyIn(BaseModel):
    email: str
    code: str


class ProfileIn(BaseModel):
    full_name: str


class ChatIn(BaseModel):
    question: str


# ----------------------------------------------------------- who is this? -- #

def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Use as Depends(get_current_user) on any endpoint that needs a signed-in user."""
    user = auth.user_for_token(_bearer_token(authorization))
    if user is None:
        raise AuthError("not_authenticated", "Your session has ended. Please sign in again.", 401)
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Use as Depends(require_admin) on endpoints only admins may call."""
    if user["role"] != "admin":
        raise AuthError("forbidden", "Only admins can do this.", 403)
    return user


# -------------------------------------------------------------- endpoints -- #

@app.post("/api/auth/register")
def register(body: RegisterIn) -> dict:
    return auth.register(body.full_name, body.email, body.password)


@app.post("/api/auth/login")
def login(body: LoginIn) -> dict:
    return auth.login(body.email, body.password)


@app.post("/api/auth/resend-code")
def resend_code(body: EmailIn) -> dict:
    return auth.resend_code(body.email)


@app.post("/api/auth/verify-otp")
def verify_otp(body: VerifyIn) -> dict:
    return auth.verify_code(body.email, body.code)


@app.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return user


@app.put("/api/auth/profile")
def update_profile(body: ProfileIn, user: dict = Depends(get_current_user)) -> dict:
    return auth.update_profile(user["id"], body.full_name)


@app.post("/api/auth/logout", status_code=204)
def logout(authorization: Optional[str] = Header(default=None)) -> Response:
    auth.logout(_bearer_token(authorization))
    return Response(status_code=204)


@app.get("/api/admin/users")
def admin_users(admin: dict = Depends(require_admin)) -> dict:
    users = auth.list_users()
    return {
        "users": users,
        "total": len(users),
        "admins": sum(1 for u in users if u["role"] == "admin"),
    }


@app.post("/api/chat")
def chat(body: ChatIn, user: dict = Depends(get_current_user)) -> dict:
    """Answer a signed-in user's question with the AcademicAgent (RAG over the NSBM documents)."""
    print(f"[chat] {user['email']} ({user['role']}) asked: {body.question[:80]!r}", flush=True)
    return chat_service.answer(body.question)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "email_mode": "smtp" if settings.sends_real_email else "terminal (SMTP_HOST not set)",
        "admin_emails_set": bool(settings.admin_emails),
        "chat": chat_service.status(),
    }

"""
Sign-in for the NSBM Admin AI portal: email + password, then a code sent by email.

How it works
    Create account  name, email and password -> we email a 6-digit code. Typing the
                    code proves the email is theirs; only then is the account saved.
    Sign in         email and password -> if they are right, we email a 6-digit code.
                    Typing the code signs them in (two-step verification).
    Roles           emails listed in ADMIN_EMAILS get the "admin" role; everyone
                    else is "user".

Passwords are stored only as PBKDF2-SHA256 hashes, codes only as keyed hashes, and
session tokens only as SHA-256 hashes.

This file is plain Python (sqlite3 + smtplib) so it can be tested without a web
server. auth_api.py puts it on the web with FastAPI.

Settings (environment variables, or a .env file in Backend/ or the project root)
    SMTP_HOST              smtp.gmail.com. Leave EMPTY while developing: no email is
                           sent and the code is printed in the uvicorn terminal.
    SMTP_PORT              587 (STARTTLS) or 465 (SSL)
    SMTP_USERNAME          the Gmail address that sends the codes
    SMTP_PASSWORD          16-character Gmail app password (not your normal password)
    SMTP_FROM              sender address (defaults to SMTP_USERNAME)
    ADMIN_EMAILS           comma-separated emails that get the admin role
    ALLOWED_EMAIL_DOMAINS  e.g. students.nsbm.ac.lk,nsbm.ac.lk  (empty = any email;
                           ADMIN_EMAILS are always allowed)
    AUTH_SECRET_KEY        long random string for hashing codes (auto-generated if empty)
    AUTH_DB_PATH           SQLite file (default: Backend/auth.db)
    SESSION_HOURS          how long a sign-in lasts (default 24)
    CORS_ORIGINS           comma-separated origins allowed to call the API (default *)
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import math
import os
import re
import secrets
import smtplib
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterator, Optional

HERE = Path(__file__).resolve().parent

CODE_TTL_SECONDS = 10 * 60      # a code works for 10 minutes
MAX_ATTEMPTS = 5                # wrong codes before the code is cancelled
RESEND_COOLDOWN_SECONDS = 30    # wait between two codes for the same email
MAX_CODES_PER_HOUR = 5          # codes per email per hour

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
PBKDF2_ITERATIONS = 600_000     # OWASP recommendation for PBKDF2-SHA256
MAX_PASSWORD_FAILURES = 5       # wrong passwords before the account is paused
LOCK_SECONDS = 15 * 60          # how long it is paused

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CODE_RE = re.compile(r"^\d{6}$")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def _load_dotenv() -> None:
    """Read Backend/.env and the project-root .env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv # type: ignore
    except ImportError:
        print("[auth] python-dotenv is not installed, so .env is NOT read. "
              "Run: python3 -m pip install python-dotenv", flush=True)
        return
    for path in (HERE / ".env", HERE.parent / ".env"):
        if path.exists():
             load_dotenv(path, override=True)


def _split(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


@dataclass
class Settings:
    db_path: str = str(HERE / "auth.db")
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    admin_emails: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    secret_key: str = ""
    session_hours: int = 24
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    password_iterations: int = PBKDF2_ITERATIONS

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        env = os.environ.get
        return cls(
            db_path=env("AUTH_DB_PATH") or str(HERE / "auth.db"),
            smtp_host=env("SMTP_HOST", "").strip(),
            smtp_port=int(env("SMTP_PORT") or 587),
            smtp_username=env("SMTP_USERNAME", "").strip(),
            # Google shows app passwords as "abcd efgh ijkl mnop"; the spaces are not part of it.
            smtp_password=env("SMTP_PASSWORD", "").replace(" ", ""),
            smtp_from=(env("SMTP_FROM") or env("SMTP_USERNAME") or "").strip(),
            admin_emails=_split(env("ADMIN_EMAILS", "")),
            allowed_domains=[d.lstrip("@") for d in _split(env("ALLOWED_EMAIL_DOMAINS", ""))],
            secret_key=env("AUTH_SECRET_KEY", "").strip(),
            session_hours=int(env("SESSION_HOURS") or 24),
            cors_origins=[o.strip() for o in (env("CORS_ORIGINS") or "*").split(",") if o.strip()],
        )

    @property
    def sends_real_email(self) -> bool:
        return bool(self.smtp_host)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class AuthError(Exception):
    """A problem the user should see. auth_api.py turns it into a JSON error."""

    def __init__(self, code: str, message: str, status: int = 400,
                 field: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.field = field
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = (stored or "").split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #

class SmtpMailer:
    """Sends the code by SMTP, or prints it when SMTP_HOST is empty."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def send_code(self, to_email: str, code: str) -> None:
        s = self.settings
        if not s.sends_real_email:
            print(
                "\n" + "=" * 60 +
                f"\n  SMTP_HOST is not set, so no email was sent."
                f"\n  Sign-in code for {to_email}: {code}\n" +
                "=" * 60 + "\n",
                flush=True,
            )
            return

        minutes = CODE_TTL_SECONDS // 60
        msg = EmailMessage()
        msg["Subject"] = f"{code} is your NSBM Admin AI verification code"
        msg["From"] = f"NSBM Admin AI <{s.smtp_from}>"
        msg["To"] = to_email
        msg.set_content(
            f"Your verification code is {code}\n\n"
            f"It expires in {minutes} minutes and can be used once.\n"
            "If you didn't try to sign in, you can ignore this email."
        )
        msg.add_alternative(f"""\
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:420px;margin:auto;padding:24px;
            background:#F8FFFE;border:1px solid #99F6E4;border-radius:12px;color:#0B3D3B">
  <h2 style="margin:0 0 4px;font-size:18px">NSBM Admin AI</h2>
  <p style="margin:0 0 20px;color:#148A64;font-size:13px">Verification code</p>
  <p style="font-size:32px;font-weight:700;letter-spacing:8px;margin:0 0 20px">{code}</p>
  <p style="font-size:14px;margin:0 0 8px">It expires in {minutes} minutes and can be used once.</p>
  <p style="font-size:12px;color:#557">If you didn't try to sign in, you can ignore this email.</p>
</div>""", subtype="html")

        if s.smtp_port == 465:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=15) as server:
                server.login(s.smtp_username, s.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as server:
                server.starttls()
                if s.smtp_username:
                    server.login(s.smtp_username, s.smtp_password)
                server.send_message(msg)


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    full_name     TEXT NOT NULL DEFAULT '',
    password_hash TEXT,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until  REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS pending_signups (
    email         TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_codes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email      TEXT NOT NULL,
    purpose    TEXT NOT NULL DEFAULT 'login',
    code_hash  TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    used       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_login_codes_email ON login_codes (email, created_at);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns added after the first version, so an existing auth.db keeps working.
MIGRATIONS = {
    "users": [
        ("password_hash", "TEXT"),
        ("failed_logins", "INTEGER NOT NULL DEFAULT 0"),
        ("locked_until", "REAL NOT NULL DEFAULT 0"),
    ],
    "login_codes": [
        ("purpose", "TEXT NOT NULL DEFAULT 'login'"),
    ],
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Sign-in logic
# --------------------------------------------------------------------------- #

class AuthService:
    def __init__(self, settings: Settings, mailer=None,
                 clock: Callable[[], float] = time.time):
        self.settings = settings
        self.mailer = mailer or SmtpMailer(settings)
        self.now = clock
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        db = self._connect()
        try:
            db.executescript(SCHEMA)
            self._migrate(db)
        finally:
            db.close()
        self._key = self._load_secret_key().encode()
        # Checked against when the email has no account, so a wrong email takes as
        # long as a wrong password and the timing doesn't reveal who has an account.
        self._dummy_hash = hash_password(secrets.token_hex(8), settings.password_iterations)

    # -- database helpers --------------------------------------------------- #

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.settings.db_path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        for table, columns in MIGRATIONS.items():
            have = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns:
                if name not in have:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. BEGIN IMMEDIATE stops two requests racing each other."""
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _get_user(self, email: str) -> Optional[sqlite3.Row]:
        db = self._connect()
        try:
            return db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        finally:
            db.close()

    def _load_secret_key(self) -> str:
        if self.settings.secret_key:
            return self.settings.secret_key
        with self._tx() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'secret_key'").fetchone()
            if row:
                return row["value"]
            key = secrets.token_hex(32)
            db.execute("INSERT INTO meta (key, value) VALUES ('secret_key', ?)", (key,))
            return key

    def _hash_code(self, email: str, code: str) -> str:
        return hmac.new(self._key, f"{email}:{code}".encode(), hashlib.sha256).hexdigest()

    # -- checks ------------------------------------------------------------- #

    def _clean_email(self, email: str) -> str:
        email = (email or "").strip().lower()
        if len(email) > 254 or not EMAIL_RE.match(email):
            raise AuthError("invalid_email", "Enter a valid email address.", 422, field="email")
        return email

    def _check_domain(self, email: str) -> None:
        allowed = self.settings.allowed_domains
        if not allowed or email in self.settings.admin_emails:
            return
        if email.rsplit("@", 1)[1] not in allowed:
            raise AuthError(
                "email_domain_not_allowed",
                "Use your NSBM email address (" + ", ".join("@" + d for d in allowed) + ").",
                403, field="email",
            )

    @staticmethod
    def _clean_name(full_name: str) -> str:
        name = " ".join((full_name or "").split())
        if not 2 <= len(name) <= 100:
            raise AuthError("invalid_name", "Enter your full name (2 to 100 characters).", 422,
                            field="full_name")
        return name

    @staticmethod
    def _check_password(password: str) -> None:
        password = password or ""
        if len(password) < PASSWORD_MIN_LENGTH:
            raise AuthError("weak_password",
                            f"Use a password of at least {PASSWORD_MIN_LENGTH} characters.",
                            422, field="password")
        if len(password) > PASSWORD_MAX_LENGTH:
            raise AuthError("weak_password",
                            f"Use a password of at most {PASSWORD_MAX_LENGTH} characters.",
                            422, field="password")

    def _role_for(self, email: str) -> str:
        return "admin" if email in self.settings.admin_emails else "user"

    @staticmethod
    def public_user(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "profile_complete": bool(row["full_name"]),
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }

    # -- 1a. create an account --------------------------------------------- #

    def register(self, full_name: str, email: str, password: str) -> dict:
        name = self._clean_name(full_name)
        email = self._clean_email(email)
        self._check_domain(email)
        self._check_password(password)

        existing = self._get_user(email)
        if existing is not None and existing["password_hash"]:
            raise AuthError("email_taken", "An account with this email already exists. Sign in instead.",
                            409, field="email")

        # The account is only saved once the emailed code is typed in (verify_code).
        password_hash = hash_password(password, self.settings.password_iterations)
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO pending_signups (email, full_name, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (email, name, password_hash, self.now()),
            )
        return self._issue_code(email, "signup")

    # -- 1b. sign in with email + password ---------------------------------- #

    def login(self, email: str, password: str) -> dict:
        email = self._clean_email(email)
        self._check_domain(email)
        now = self.now()
        user = self._get_user(email)

        if user is not None and user["locked_until"] > now:
            wait = math.ceil(user["locked_until"] - now)
            raise AuthError(
                "too_many_attempts",
                f"Too many wrong passwords. Try again in {math.ceil(wait / 60)} minutes.",
                429, retry_after=wait,
            )

        stored = user["password_hash"] if user is not None else None
        correct = verify_password(password or "", stored or self._dummy_hash) and bool(stored)

        if not correct:
            failure = AuthError("invalid_credentials", "Incorrect email or password.", 401, field="password")
            if user is not None:
                with self._tx() as db:
                    failures = db.execute("SELECT failed_logins FROM users WHERE id = ?",
                                          (user["id"],)).fetchone()["failed_logins"] + 1
                    if failures >= MAX_PASSWORD_FAILURES:
                        db.execute("UPDATE users SET failed_logins = 0, locked_until = ? WHERE id = ?",
                                   (now + LOCK_SECONDS, user["id"]))
                        failure = AuthError(
                            "too_many_attempts",
                            f"Too many wrong passwords. Try again in {LOCK_SECONDS // 60} minutes.",
                            429, retry_after=LOCK_SECONDS,
                        )
                    else:
                        db.execute("UPDATE users SET failed_logins = ? WHERE id = ?", (failures, user["id"]))
            raise failure

        if user["failed_logins"]:
            with self._tx() as db:
                db.execute("UPDATE users SET failed_logins = 0 WHERE id = ?", (user["id"],))
        return self._issue_code(email, "login")

    # -- sending codes ------------------------------------------------------ #

    def resend_code(self, email: str) -> dict:
        """A new code for a sign-in or sign-up that passed step 1 in the last 10 minutes."""
        email = self._clean_email(email)
        now = self.now()
        db = self._connect()
        try:
            row = db.execute(
                "SELECT * FROM login_codes WHERE email = ? ORDER BY created_at DESC LIMIT 1", (email,)
            ).fetchone()
        finally:
            db.close()
        locked = row is not None and row["attempts"] >= MAX_ATTEMPTS
        if (row is None or now - row["created_at"] > CODE_TTL_SECONDS
                or (row["used"] and not locked)):
            raise AuthError("otp_missing", "Please sign in again to get a new code.")
        return self._issue_code(email, row["purpose"])

    def _issue_code(self, email: str, purpose: str) -> dict:
        now = self.now()
        with self._tx() as db:
            recent = db.execute(
                "SELECT created_at FROM login_codes WHERE email = ? AND created_at > ? "
                "ORDER BY created_at",
                (email, now - 3600),
            ).fetchall()
            # The 30-second wait only applies while the last code can still be used.
            active = db.execute(
                "SELECT created_at FROM login_codes WHERE email = ? AND used = 0 AND expires_at > ? "
                "ORDER BY created_at DESC LIMIT 1",
                (email, now),
            ).fetchone()
            if active:
                elapsed = now - active["created_at"]
                if elapsed < RESEND_COOLDOWN_SECONDS:
                    wait = math.ceil(RESEND_COOLDOWN_SECONDS - elapsed)
                    raise AuthError(
                        "otp_cooldown",
                        f"Please wait {wait} seconds before asking for a new code.",
                        429, retry_after=wait,
                    )
            if len(recent) >= MAX_CODES_PER_HOUR:
                wait = math.ceil(recent[0]["created_at"] + 3600 - now)
                raise AuthError(
                    "otp_rate_limited",
                    f"Too many codes for this email. Try again in {math.ceil(wait / 60)} minutes.",
                    429, retry_after=wait,
                )

            # Only the newest code works.
            db.execute("UPDATE login_codes SET used = 1 WHERE email = ? AND used = 0", (email,))
            code = f"{secrets.randbelow(1_000_000):06d}"
            cur = db.execute(
                "INSERT INTO login_codes (email, purpose, code_hash, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (email, purpose, self._hash_code(email, code), now, now + CODE_TTL_SECONDS),
            )
            code_id = cur.lastrowid

        try:
            self.mailer.send_code(email, code)
        except Exception as exc:  # wrong app password, no internet, ...
            with self._tx() as db:
                db.execute("DELETE FROM login_codes WHERE id = ?", (code_id,))
            print(f"[auth] Could not send the code to {email}: {exc!r}", flush=True)
            raise AuthError(
                "email_failed",
                "We couldn't send the email. Please try again in a moment.",
                502,
            ) from exc

        return {
            "sent": True,
            "expires_in": CODE_TTL_SECONDS,
            "resend_after": RESEND_COOLDOWN_SECONDS,
        }

    # -- 2. check the code and sign in ------------------------------------- #

    def verify_code(self, email: str, code: str) -> dict:
        email = self._clean_email(email)
        code = (code or "").strip().replace(" ", "")
        if not CODE_RE.match(code):
            raise AuthError("otp_invalid", "Enter the 6-digit code from the email.", 422, field="code")
        now = self.now()
        failure: Optional[AuthError] = None
        user = None
        is_new_user = False

        with self._tx() as db:
            row = db.execute(
                "SELECT * FROM login_codes WHERE email = ? AND used = 0 "
                "ORDER BY created_at DESC LIMIT 1",
                (email,),
            ).fetchone()

            if row is None:
                failure = AuthError("otp_missing", "There's no active code for this email. Please sign in again.")
            elif now > row["expires_at"]:
                db.execute("UPDATE login_codes SET used = 1 WHERE id = ?", (row["id"],))
                failure = AuthError("otp_expired", "That code has expired. Ask for a new one.")
            elif not hmac.compare_digest(row["code_hash"], self._hash_code(email, code)):
                attempts = row["attempts"] + 1
                left = MAX_ATTEMPTS - attempts
                if left <= 0:
                    db.execute("UPDATE login_codes SET attempts = ?, used = 1 WHERE id = ?",
                               (attempts, row["id"]))
                    failure = AuthError("otp_locked", "Too many wrong codes. Ask for a new one.")
                else:
                    db.execute("UPDATE login_codes SET attempts = ? WHERE id = ?",
                               (attempts, row["id"]))
                    failure = AuthError(
                        "otp_invalid",
                        f"That code is incorrect. You have {left} attempt{'s' if left > 1 else ''} left.",
                        field="code",
                    )
            else:
                # Correct code: use it up.
                db.execute("UPDATE login_codes SET used = 1 WHERE id = ?", (row["id"],))
                role = self._role_for(email)
                existing = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

                if row["purpose"] == "signup":
                    pending = db.execute("SELECT * FROM pending_signups WHERE email = ?", (email,)).fetchone()
                    if pending is None:
                        failure = AuthError("signup_missing", "Please create your account again.")
                    elif existing is None:
                        db.execute(
                            "INSERT INTO users (email, full_name, password_hash, role, created_at, last_login_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (email, pending["full_name"], pending["password_hash"], role, _iso(now), _iso(now)),
                        )
                        is_new_user = True
                    else:
                        db.execute(
                            "UPDATE users SET full_name = ?, password_hash = ?, role = ?, last_login_at = ?, "
                            "failed_logins = 0, locked_until = 0 WHERE id = ?",
                            (pending["full_name"], pending["password_hash"], role, _iso(now), existing["id"]),
                        )
                        is_new_user = True
                    db.execute("DELETE FROM pending_signups WHERE email = ?", (email,))
                elif existing is None or not existing["password_hash"]:
                    failure = AuthError("otp_missing", "Please sign in again.")
                else:
                    db.execute("UPDATE users SET role = ?, last_login_at = ? WHERE id = ?",
                               (role, _iso(now), existing["id"]))

                if failure is None:
                    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
                    token = secrets.token_urlsafe(32)
                    expires_at = now + self.settings.session_hours * 3600
                    db.execute(
                        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                        (_sha256(token), user["id"], now, expires_at),
                    )
                    # Housekeeping: forget old codes, unfinished sign-ups and expired sessions.
                    db.execute("DELETE FROM login_codes WHERE created_at < ?", (now - 86400,))
                    db.execute("DELETE FROM pending_signups WHERE created_at < ?", (now - 86400,))
                    db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))

        # Raised after COMMIT so a wrong guess still counts as an attempt.
        if failure:
            raise failure

        return {
            "token": token,
            "token_type": "bearer",
            "expires_at": _iso(expires_at),
            "is_new_user": is_new_user,
            "user": self.public_user(user),
        }

    # -- sessions and profile ---------------------------------------------- #

    def user_for_token(self, token: Optional[str]) -> Optional[dict]:
        if not token:
            return None
        db = self._connect()
        try:
            row = db.execute(
                "SELECT u.*, s.expires_at AS session_expires FROM sessions s "
                "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
                (_sha256(token),),
            ).fetchone()
            if row is None:
                return None
            if self.now() > row["session_expires"]:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha256(token),))
                return None
            return self.public_user(row)
        finally:
            db.close()

    def update_profile(self, user_id: int, full_name: str) -> dict:
        name = self._clean_name(full_name)
        with self._tx() as db:
            db.execute("UPDATE users SET full_name = ? WHERE id = ?", (name, user_id))
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self.public_user(row)

    def logout(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._tx() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha256(token),))

    # -- admin -------------------------------------------------------------- #

    def list_users(self) -> list[dict]:
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT * FROM users WHERE password_hash IS NOT NULL ORDER BY role, created_at"
            ).fetchall()
            return [self.public_user(r) for r in rows]
        finally:
            db.close()
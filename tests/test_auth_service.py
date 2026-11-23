"""Tests for Backend/auth_service.py.  Run from the project folder:  python3 -m pytest tests/test_auth_service.py"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Backend"))

from auth_service import (  # noqa: E402
    CODE_TTL_SECONDS,
    LOCK_SECONDS,
    MAX_ATTEMPTS,
    MAX_PASSWORD_FAILURES,
    RESEND_COOLDOWN_SECONDS,
    AuthError,
    AuthService,
    Settings,
    hash_password,
    verify_password,
)

PASSWORD = "correct horse 42"


class FakeMailer:
    def __init__(self):
        self.sent = []          # (email, code)
        self.fail = False

    def send_code(self, to_email, code):
        if self.fail:
            raise OSError("SMTP down")
        self.sent.append((to_email, code))

    def last_code(self):
        return self.sent[-1][1]


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "auth.db"),
        admin_emails=["esalawathsilu@gmail.com"],
        secret_key="test-secret",
        password_iterations=1_000,   # fast for tests; the real default is 600,000
    )
    mailer, clock = FakeMailer(), Clock()
    return AuthService(settings, mailer=mailer, clock=clock), mailer, clock


def create_account(service, mailer, email, name="Test User", password=PASSWORD):
    service.register(name, email, password)
    return service.verify_code(email, mailer.last_code())


def sign_in(service, mailer, email, password=PASSWORD):
    service.login(email, password)
    return service.verify_code(email, mailer.last_code())


def wrong(code):
    return "000000" if code != "000000" else "111111"


# ---------------------------------------------------------------- accounts -- #

def test_create_account_needs_the_email_code(env):
    service, mailer, _ = env
    sent = service.register("  Demini   De Alwis ", "  Student@Students.NSBM.ac.lk ", PASSWORD)
    assert sent == {"sent": True, "expires_in": CODE_TTL_SECONDS, "resend_after": RESEND_COOLDOWN_SECONDS}
    assert mailer.sent[0][0] == "student@students.nsbm.ac.lk"
    # Not an account until the code is typed in.
    assert service.list_users() == []

    result = service.verify_code("student@students.nsbm.ac.lk", mailer.last_code())
    assert result["is_new_user"] is True
    assert result["user"]["full_name"] == "Demini De Alwis"
    assert result["user"]["role"] == "user"
    assert service.user_for_token(result["token"])["email"] == "student@students.nsbm.ac.lk"


def test_existing_account_cannot_register_again(env):
    service, mailer, _ = env
    create_account(service, mailer, "a@nsbm.ac.lk")
    with pytest.raises(AuthError) as err:
        service.register("Someone Else", "a@nsbm.ac.lk", "another password")
    assert err.value.code == "email_taken" and err.value.status == 409


def test_unverified_signup_can_be_redone(env):
    service, mailer, clock = env
    service.register("First Try", "a@nsbm.ac.lk", "first password")
    clock.advance(CODE_TTL_SECONDS + 1)         # never typed the code
    created = create_account(service, mailer, "a@nsbm.ac.lk", name="Second Try", password="second password")
    assert created["user"]["full_name"] == "Second Try"
    clock.advance(RESEND_COOLDOWN_SECONDS + 1)
    with pytest.raises(AuthError):
        service.login("a@nsbm.ac.lk", "first password")
    assert service.login("a@nsbm.ac.lk", "second password")["sent"]


@pytest.mark.parametrize("password", ["", "short", "x" * 129])
def test_password_length(env, password):
    service, _, _ = env
    with pytest.raises(AuthError) as err:
        service.register("Test User", "a@nsbm.ac.lk", password)
    assert err.value.code == "weak_password" and err.value.field == "password"


def test_bad_name(env):
    service, _, _ = env
    with pytest.raises(AuthError) as err:
        service.register(" x ", "a@nsbm.ac.lk", PASSWORD)
    assert err.value.code == "invalid_name"


# ----------------------------------------------------------------- sign in -- #

def test_sign_in_is_password_then_code(env):
    service, mailer, clock = env
    created = create_account(service, mailer, "a@nsbm.ac.lk")
    clock.advance(RESEND_COOLDOWN_SECONDS + 1)
    sent = service.login("A@nsbm.ac.lk", PASSWORD)
    assert sent["sent"] and mailer.sent[-1][0] == "a@nsbm.ac.lk"
    result = service.verify_code("a@nsbm.ac.lk", mailer.last_code())
    assert result["is_new_user"] is False
    assert result["user"]["id"] == created["user"]["id"]


def test_wrong_password_sends_no_code(env):
    service, mailer, clock = env
    create_account(service, mailer, "a@nsbm.ac.lk")
    clock.advance(RESEND_COOLDOWN_SECONDS + 1)
    before = len(mailer.sent)
    with pytest.raises(AuthError) as err:
        service.login("a@nsbm.ac.lk", "wrong password")
    assert err.value.code == "invalid_credentials" and err.value.status == 401
    assert len(mailer.sent) == before


def test_unknown_email_gets_the_same_message(env):
    service, _, _ = env
    with pytest.raises(AuthError) as err:
        service.login("nobody@nsbm.ac.lk", PASSWORD)
    assert err.value.code == "invalid_credentials"
    assert err.value.message == "Incorrect email or password."


def test_too_many_wrong_passwords_pauses_the_account(env):
    service, mailer, clock = env
    create_account(service, mailer, "a@nsbm.ac.lk")
    for _ in range(MAX_PASSWORD_FAILURES - 1):
        with pytest.raises(AuthError) as err:
            service.login("a@nsbm.ac.lk", "wrong password")
        assert err.value.code == "invalid_credentials"
    with pytest.raises(AuthError) as err:
        service.login("a@nsbm.ac.lk", "wrong password")
    assert err.value.code == "too_many_attempts" and err.value.status == 429

    # Even the right password waits until the pause is over.
    with pytest.raises(AuthError) as err:
        service.login("a@nsbm.ac.lk", PASSWORD)
    assert err.value.code == "too_many_attempts"
    clock.advance(LOCK_SECONDS + 1)
    assert service.login("a@nsbm.ac.lk", PASSWORD)["sent"]


def test_admin_email_gets_admin_role(env):
    service, mailer, clock = env
    create_account(service, mailer, "student@nsbm.ac.lk")
    admin = create_account(service, mailer, "EsalaWathsilu@gmail.com", name="Esala Kumarapperuma")
    assert admin["user"]["role"] == "admin"
    assert service.user_for_token(admin["token"])["role"] == "admin"
    roles = {u["email"]: u["role"] for u in service.list_users()}
    assert roles == {"student@nsbm.ac.lk": "user", "esalawathsilu@gmail.com": "admin"}

    # Removing the email from ADMIN_EMAILS demotes it at the next sign-in.
    service.settings.admin_emails = []
    clock.advance(RESEND_COOLDOWN_SECONDS + 1)
    assert sign_in(service, mailer, "esalawathsilu@gmail.com")["user"]["role"] == "user"


def test_allowed_domains(env):
    service, mailer, _ = env
    service.settings.allowed_domains = ["students.nsbm.ac.lk", "nsbm.ac.lk"]
    service.register("Test User", "someone@students.nsbm.ac.lk", PASSWORD)
    with pytest.raises(AuthError) as err:
        service.register("Test User", "someone@gmail.com", PASSWORD)
    assert err.value.code == "email_domain_not_allowed" and err.value.status == 403
    # Admin emails are allowed even outside the NSBM domains.
    service.register("Esala", "esalawathsilu@gmail.com", PASSWORD)
    assert mailer.sent[-1][0] == "esalawathsilu@gmail.com"


# ------------------------------------------------------------------- codes -- #

def test_wrong_codes_count_down_then_lock(env):
    service, mailer, _ = env
    create_account(service, mailer, "a@nsbm.ac.lk")
    service.logout(None)
    service.login("a@nsbm.ac.lk", PASSWORD)
    good = mailer.last_code()

    for left in range(MAX_ATTEMPTS - 1, 0, -1):
        with pytest.raises(AuthError) as err:
            service.verify_code("a@nsbm.ac.lk", wrong(good))
        assert err.value.code == "otp_invalid" and f"{left} attempt" in err.value.message
    with pytest.raises(AuthError) as err:
        service.verify_code("a@nsbm.ac.lk", wrong(good))
    assert err.value.code == "otp_locked"
    with pytest.raises(AuthError) as err:
        service.verify_code("a@nsbm.ac.lk", good)
    assert err.value.code == "otp_missing"

    # A locked code can be replaced with "send a new code" (no password needed again).
    service.resend_code("a@nsbm.ac.lk")
    assert service.verify_code("a@nsbm.ac.lk", mailer.last_code())["token"]


def test_code_expires_after_ten_minutes(env):
    service, mailer, clock = env
    service.register("Test User", "a@nsbm.ac.lk", PASSWORD)
    clock.advance(CODE_TTL_SECONDS + 1)
    with pytest.raises(AuthError) as err:
        service.verify_code("a@nsbm.ac.lk", mailer.last_code())
    assert err.value.code == "otp_expired"


def test_code_works_only_once(env):
    service, mailer, _ = env
    service.register("Test User", "a@nsbm.ac.lk", PASSWORD)
    code = mailer.last_code()
    service.verify_code("a@nsbm.ac.lk", code)
    with pytest.raises(AuthError) as err:
        service.verify_code("a@nsbm.ac.lk", code)
    assert err.value.code == "otp_missing"


def test_resend_cooldown_and_only_newest_code_works(env):
    service, mailer, clock = env
    create_account(service, mailer, "a@nsbm.ac.lk")
    service.login("a@nsbm.ac.lk", PASSWORD)
    old = mailer.last_code()

    clock.advance(10)
    with pytest.raises(AuthError) as err:
        service.resend_code("a@nsbm.ac.lk")
    assert err.value.code == "otp_cooldown" and err.value.retry_after == RESEND_COOLDOWN_SECONDS - 10

    clock.advance(RESEND_COOLDOWN_SECONDS)
    service.resend_code("a@nsbm.ac.lk")
    new = mailer.last_code()
    if old != new:
        with pytest.raises(AuthError):
            service.verify_code("a@nsbm.ac.lk", old)
    assert service.verify_code("a@nsbm.ac.lk", new)["is_new_user"] is False


def test_resend_needs_a_recent_password_step(env):
    service, mailer, clock = env
    # No sign-in started: nothing to resend, and nothing is emailed.
    with pytest.raises(AuthError) as err:
        service.resend_code("a@nsbm.ac.lk")
    assert err.value.code == "otp_missing"
    # After a finished sign-in, resend does not hand out another code.
    create_account(service, mailer, "a@nsbm.ac.lk")
    before = len(mailer.sent)
    with pytest.raises(AuthError):
        service.resend_code("a@nsbm.ac.lk")
    # After 10 minutes, the password is needed again.
    service.login("a@nsbm.ac.lk", PASSWORD)
    clock.advance(CODE_TTL_SECONDS + 1)
    with pytest.raises(AuthError):
        service.resend_code("a@nsbm.ac.lk")
    assert len(mailer.sent) == before + 1


def test_no_wait_after_a_code_was_used(env):
    service, mailer, _ = env
    first = create_account(service, mailer, "a@nsbm.ac.lk")
    service.logout(first["token"])
    assert sign_in(service, mailer, "a@nsbm.ac.lk")["token"]


def test_hourly_limit(env):
    service, _, clock = env
    for _ in range(5):
        service.register("Test User", "a@nsbm.ac.lk", PASSWORD)
        clock.advance(RESEND_COOLDOWN_SECONDS + 1)
    with pytest.raises(AuthError) as err:
        service.register("Test User", "a@nsbm.ac.lk", PASSWORD)
    assert err.value.code == "otp_rate_limited"
    clock.advance(3600)
    assert service.register("Test User", "a@nsbm.ac.lk", PASSWORD)["sent"]


@pytest.mark.parametrize("code", ["", "12345", "1234567", "abcdef"])
def test_bad_code_format(env, code):
    service, _, _ = env
    with pytest.raises(AuthError) as err:
        service.verify_code("a@nsbm.ac.lk", code)
    assert err.value.code == "otp_invalid" and err.value.status == 422


@pytest.mark.parametrize("email", ["", "not-an-email", "a@b", "a b@nsbm.ac.lk"])
def test_bad_email(env, email):
    service, _, _ = env
    with pytest.raises(AuthError) as err:
        service.login(email, PASSWORD)
    assert err.value.code == "invalid_email" and err.value.field == "email"


def test_email_failure_does_not_start_cooldown(env):
    service, mailer, _ = env
    mailer.fail = True
    with pytest.raises(AuthError) as err:
        service.register("Test User", "a@nsbm.ac.lk", PASSWORD)
    assert err.value.code == "email_failed" and err.value.status == 502
    mailer.fail = False
    assert service.register("Test User", "a@nsbm.ac.lk", PASSWORD)["sent"]


# ---------------------------------------------------------------- sessions -- #

def test_logout_and_session_expiry(env):
    service, mailer, clock = env
    a = create_account(service, mailer, "a@nsbm.ac.lk")
    service.logout(a["token"])
    assert service.user_for_token(a["token"]) is None

    b = create_account(service, mailer, "b@nsbm.ac.lk")
    clock.advance(service.settings.session_hours * 3600 + 1)
    assert service.user_for_token(b["token"]) is None
    assert service.user_for_token("made-up-token") is None
    assert service.user_for_token(None) is None


# ---------------------------------------------------------------- storage -- #

def test_passwords_codes_and_tokens_are_not_stored_in_plain_text(env):
    service, mailer, _ = env
    result = create_account(service, mailer, "a@nsbm.ac.lk")
    db = sqlite3.connect(service.settings.db_path)
    dump = "\n".join(db.iterdump())
    db.close()
    assert PASSWORD not in dump
    assert mailer.last_code() not in dump
    assert result["token"] not in dump


def test_password_hashing():
    stored = hash_password("s3cret pass", iterations=1_000)
    assert stored.startswith("pbkdf2_sha256$1000$")
    assert verify_password("s3cret pass", stored)
    assert not verify_password("s3cret pasS", stored)
    assert not verify_password("anything", None)
    assert not verify_password("anything", "not-a-hash")


def test_old_database_is_upgraded(tmp_path):
    """An auth.db made by the email-code-only version keeps working."""
    path = tmp_path / "auth.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL DEFAULT '', role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL, last_login_at TEXT);
        CREATE TABLE login_codes (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL,
            code_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, used INTEGER NOT NULL DEFAULT 0);
        INSERT INTO users (email, full_name, role, created_at) VALUES ('old@nsbm.ac.lk', 'Old User', 'user', '2026-11-16');
    """)
    db.commit()
    db.close()

    mailer = FakeMailer()
    service = AuthService(Settings(db_path=str(path), password_iterations=1_000), mailer=mailer)
    # The old account has no password, so it signs up once to set one.
    with pytest.raises(AuthError):
        service.login("old@nsbm.ac.lk", PASSWORD)
    result = create_account(service, mailer, "old@nsbm.ac.lk", name="Old User")
    assert result["user"]["email"] == "old@nsbm.ac.lk"


def test_secret_key_is_generated_once_and_reused(tmp_path):
    settings = Settings(db_path=str(tmp_path / "auth.db"), password_iterations=1_000)
    first = AuthService(settings, mailer=FakeMailer())
    second = AuthService(settings, mailer=FakeMailer())
    assert first._key == second._key and len(first._key) == 64
"""Per-browser Google OAuth connection management for Nexa."""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from Backend.LLMProvider import get_config
from Backend.MongoStore import current_chat_user_id
from Backend.Paths import DATA_DIR


ROOT = Path(__file__).resolve().parent
DATA_PATH = DATA_DIR / "GoogleConnections.json"
STATE_PATH = DATA_DIR / "GoogleOAuthStates.json"
SESSION_COOKIE = "nexa_google_session"
_SESSION_ID = contextvars.ContextVar("nexa_google_session_id", default="")
_USER_ID = contextvars.ContextVar("nexa_google_user_id", default="")
_JSON_LOCK = threading.RLock()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GMAIL_PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

SERVICES = {
    "gmail": {
        "label": "Gmail",
        "scopes": [
            "openid",
            "email",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ],
    },
    "google_calendar": {
        "label": "Google Calendar",
        "scopes": [
            "openid",
            "email",
            "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
            "https://www.googleapis.com/auth/calendar.events.freebusy",
            "https://www.googleapis.com/auth/calendar.events",
        ],
    },
    "google_drive": {
        "label": "Google Drive",
        "scopes": [
            "openid",
            "email",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    },
}


class GoogleOAuthError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, fallback: Any) -> Any:
    with _JSON_LOCK:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback


def _write_json(path: Path, payload: Any) -> None:
    with _JSON_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.{secrets.token_hex(8)}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)


def _client_id() -> str:
    return get_config("GOOGLE_OAUTH_CLIENT_ID", "").strip()


def _client_secret() -> str:
    return get_config("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()


def _redirect_uri() -> str:
    default_base_url = get_config("APP_BASE_URL", "http://localhost:8000").rstrip("/")
    return get_config("GOOGLE_OAUTH_REDIRECT_URI", f"{default_base_url}/api/google/oauth/callback").strip()


def google_oauth_redirect_uri() -> str:
    return _redirect_uri()


def _fernet_key() -> bytes:
    raw = get_config("GOOGLE_TOKEN_ENCRYPTION_KEY", "").strip()
    if not raw:
        raise GoogleOAuthError(
            "Google token encryption is not configured. Set GOOGLE_TOKEN_ENCRYPTION_KEY in .env."
        )
    return raw.encode("utf-8")


def _cipher() -> Any:
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise GoogleOAuthError(
            "Google OAuth requires the cryptography package. Install requirements.txt first."
        ) from exc
    try:
        return Fernet(_fernet_key())
    except ValueError as exc:
        raise GoogleOAuthError("GOOGLE_TOKEN_ENCRYPTION_KEY is not a valid Fernet key.") from exc


def google_oauth_is_configured() -> bool:
    try:
        _cipher()
    except GoogleOAuthError:
        return False
    return bool(_client_id() and _client_secret() and _redirect_uri())


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def current_session_id() -> str:
    return _SESSION_ID.get()


class google_session_context:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.token: contextvars.Token[str] | None = None

    def __enter__(self) -> "google_session_context":
        self.token = _SESSION_ID.set(self.session_id)
        return self

    def __exit__(self, *_: object) -> None:
        if self.token is not None:
            _SESSION_ID.reset(self.token)


class google_user_context:
    """Bind a delegated Google connection to an authenticated Nexa user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.token: contextvars.Token[str] | None = None

    def __enter__(self) -> "google_user_context":
        self.token = _USER_ID.set(self.user_id)
        return self

    def __exit__(self, *_: object) -> None:
        if self.token is not None:
            _USER_ID.reset(self.token)


def _active_user_id(user_id: str = "") -> str:
    return user_id or _USER_ID.get() or current_chat_user_id()


def _encrypted_token(tokens: dict[str, Any]) -> str:
    return _cipher().encrypt(json.dumps(tokens).encode("utf-8")).decode("utf-8")


def _decrypted_token(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("token")
    if not isinstance(payload, str) or not payload:
        raise GoogleOAuthError("The saved Google connection is invalid. Connect the app again.")
    try:
        return json.loads(_cipher().decrypt(payload.encode("utf-8")).decode("utf-8"))
    except Exception as exc:
        raise GoogleOAuthError("The saved Google connection cannot be decrypted. Connect the app again.") from exc


def _records() -> list[dict[str, Any]]:
    data = _read_json(DATA_PATH, [])
    return data if isinstance(data, list) else []


def _save_records(records: list[dict[str, Any]]) -> None:
    _write_json(DATA_PATH, records)


def _record_index(
    user_id: str,
    session_id: str,
    service: str,
    records: list[dict[str, Any]] | None = None,
) -> int | None:
    values = records if records is not None else _records()
    for index, item in enumerate(values):
        if user_id and item.get("user_id") == user_id and item.get("service") == service:
            return index
    # Old records were browser-cookie scoped. Only migrate a record that has
    # no owner and is presented by the same browser session.
    if user_id:
        for index, item in enumerate(values):
            if not item.get("user_id") and item.get("session_id") == session_id and item.get("service") == service:
                return index
    elif session_id:
        for index, item in enumerate(values):
            if item.get("session_id") == session_id and item.get("service") == service:
                return index
    return None


def _save_token(session_id: str, user_id: str, service: str, token: dict[str, Any], email: str = "") -> None:
    with _JSON_LOCK:
        records = _records()
        record = {
            "user_id": user_id,
            "session_id": session_id,
            "service": service,
            "email": email,
            "token": _encrypted_token(token),
            "updated_at": _now().isoformat(),
        }
        existing = _record_index(user_id, session_id, service, records)
        if existing is None:
            records.append(record)
        else:
            records[existing] = record
        _save_records(records)


def _get_record(session_id: str, service: str, user_id: str = "") -> dict[str, Any] | None:
    active_user_id = _active_user_id(user_id)
    with _JSON_LOCK:
        records = _records()
        index = _record_index(active_user_id, session_id, service, records)
        if index is None:
            return None
        record = records[index]
        # One-time, safe migration of a legacy record requires possession of
        # the original browser cookie and an authenticated user identity.
        if active_user_id and not record.get("user_id"):
            record = {**record, "user_id": active_user_id, "updated_at": _now().isoformat()}
            records[index] = record
            _save_records(records)
        return record


def service_status(session_id: str, user_id: str = "") -> list[dict[str, Any]]:
    statuses = []
    for service, info in SERVICES.items():
        record = _get_record(session_id, service, user_id) if (session_id or user_id) else None
        statuses.append({
            "service": service,
            "label": info["label"],
            "configured": google_oauth_is_configured(),
            "connected": bool(record),
            "email": str(record.get("email") or "") if record else "",
        })
    return statuses


def start_authorization(service: str, session_id: str, user_id: str) -> str:
    if service not in SERVICES:
        raise GoogleOAuthError("Unknown Google service.")
    if not user_id:
        raise GoogleOAuthError("Sign in to Nexa before connecting a Google service.")
    if not google_oauth_is_configured():
        raise GoogleOAuthError(
            "Google OAuth is not configured. Add the client ID, client secret, redirect URI, and token key to .env."
        )
    state = secrets.token_urlsafe(32)
    with _JSON_LOCK:
        states = _read_json(STATE_PATH, [])
        if not isinstance(states, list):
            states = []
        cutoff = _now() - timedelta(minutes=15)
        states = [
            item
            for item in states
            if isinstance(item, dict)
            and item.get("created_at", "") >= cutoff.isoformat()
        ]
        states.append({
            "state": state,
            "service": service,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": _now().isoformat(),
        })
        _write_json(STATE_PATH, states)
    query = urlencode({
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": " ".join(SERVICES[service]["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    })
    return f"{GOOGLE_AUTH_URL}?{query}"


def complete_authorization(code: str, state: str) -> tuple[str, str]:
    with _JSON_LOCK:
        states = _read_json(STATE_PATH, [])
        states = states if isinstance(states, list) else []
        match = next(
            (
                item
                for item in states
                if isinstance(item, dict)
                and secrets.compare_digest(str(item.get("state", "")), state)
            ),
            None,
        )
        _write_json(STATE_PATH, [item for item in states if item is not match])
    if not match:
        raise GoogleOAuthError("This Google connection link is invalid or has expired. Try again.")
    if not code:
        raise GoogleOAuthError("Google did not return an authorization code.")
    try:
        created_at = datetime.fromisoformat(str(match.get("created_at") or ""))
    except ValueError as exc:
        raise GoogleOAuthError("This Google connection link is invalid. Try again.") from exc
    if created_at < _now() - timedelta(minutes=15):
        raise GoogleOAuthError("This Google connection link has expired. Try again.")

    try:
        response = requests.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "redirect_uri": _redirect_uri(),
            "grant_type": "authorization_code",
        }, timeout=30)
    except requests.RequestException as exc:
        raise GoogleOAuthError(
            "Google authorization could not be reached. Check the internet connection and try again."
        ) from exc
    if not response.ok:
        raise GoogleOAuthError("Google could not authorize this account. Check your OAuth client and redirect URI.")
    try:
        token = response.json()
    except ValueError as exc:
        raise GoogleOAuthError("Google returned an invalid authorization response.") from exc
    token["expires_at"] = int((_now() + timedelta(seconds=int(token.get("expires_in", 3600)))).timestamp())
    email = ""
    access_token = str(token.get("access_token") or "")
    if access_token:
        try:
            profile = requests.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15,
            )
            if profile.ok:
                email = str(profile.json().get("email") or "")
        except (requests.RequestException, ValueError):
            email = ""
        if not email and str(match.get("service")) == "gmail":
            try:
                gmail_profile = requests.get(
                    GMAIL_PROFILE_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15,
                )
                if gmail_profile.ok:
                    email = str(gmail_profile.json().get("emailAddress") or "")
            except (requests.RequestException, ValueError):
                email = ""
    _save_token(
        str(match["session_id"]),
        str(match.get("user_id") or ""),
        str(match["service"]),
        token,
        email,
    )
    return str(match["service"]), email


def _access_token(session_id: str, service: str, user_id: str = "") -> str:
    active_user_id = _active_user_id(user_id)
    record = _get_record(session_id, service, active_user_id)
    if not record:
        raise GoogleOAuthError(f"Connect {SERVICES[service]['label']} before using it.")
    token = _decrypted_token(record)
    if int(token.get("expires_at", 0)) <= int((_now() + timedelta(seconds=60)).timestamp()):
        refresh_token = str(token.get("refresh_token") or "")
        if not refresh_token:
            raise GoogleOAuthError("Google access expired. Connect this service again.")
        try:
            refreshed = requests.post(GOOGLE_TOKEN_URL, data={
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }, timeout=30)
        except requests.RequestException as exc:
            raise GoogleOAuthError(
                "Google access could not be refreshed. Check the internet connection and try again."
            ) from exc
        if not refreshed.ok:
            raise GoogleOAuthError("Google access expired. Connect this service again.")
        try:
            new_token = refreshed.json()
        except ValueError as exc:
            raise GoogleOAuthError("Google returned an invalid refresh response.") from exc
        token.update(new_token)
        token["refresh_token"] = token.get("refresh_token") or refresh_token
        token["expires_at"] = int((_now() + timedelta(seconds=int(token.get("expires_in", 3600)))).timestamp())
        _save_token(session_id, active_user_id, service, token, str(record.get("email") or ""))
    access_token = str(token.get("access_token") or "")
    if not access_token:
        raise GoogleOAuthError("Google did not provide an access token. Connect this service again.")
    return access_token


def google_mcp_header(service: str) -> str:
    session_id = current_session_id()
    if (not session_id and not _active_user_id()) or service not in SERVICES:
        return ""
    return f"Bearer {_access_token(session_id, service)}"


def google_access_token(service: str) -> str:
    """Return an access token for the current browser session and service."""
    session_id = current_session_id()
    if not session_id and not _active_user_id():
        raise GoogleOAuthError("No browser session is available. Reconnect the Google service.")
    if service not in SERVICES:
        raise GoogleOAuthError("Unknown Google service.")
    return _access_token(session_id, service)


def google_connected_email(service: str) -> str:
    """Return the connected account address for the current session."""
    session_id = current_session_id()
    record = _get_record(session_id, service) if service in SERVICES else None
    return str(record.get("email") or "") if record else ""


def google_mcp_connected(service: str) -> bool:
    session_id = current_session_id()
    return bool(service in SERVICES and _get_record(session_id, service))


def google_connection_signature() -> str:
    session_id = current_session_id()
    if not session_id:
        return "anonymous"
    pieces = [session_id]
    for service in SERVICES:
        try:
            header = google_mcp_header(service)
        except GoogleOAuthError:
            header = ""
        pieces.append(hashlib.sha256(header.encode("utf-8")).hexdigest() if header else "")
    return ":".join(pieces)


def disconnect_service(session_id: str, service: str, user_id: str = "") -> None:
    if service not in SERVICES:
        raise GoogleOAuthError("Unknown Google service.")
    with _JSON_LOCK:
        records = _records()
        retained = []
        active_user_id = _active_user_id(user_id)
        for record in records:
            owned = (
                bool(active_user_id and record.get("user_id") == active_user_id)
                or (not record.get("user_id") and record.get("session_id") == session_id)
            )
            if owned and record.get("service") == service:
                try:
                    token = _decrypted_token(record)
                    refresh = str(token.get("refresh_token") or token.get("access_token") or "")
                    if refresh:
                        requests.post(GOOGLE_REVOKE_URL, params={"token": refresh}, timeout=10)
                except Exception:
                    pass
                continue
            retained.append(record)
        _save_records(retained)

"""MongoDB-backed accounts and per-user Nexa chat sessions."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError


_chat_session_id: ContextVar[str] = ContextVar("nexa_chat_session_id", default="")
_chat_user_id: ContextVar[str] = ContextVar("nexa_chat_user_id", default="")
_client: MongoClient | None = None
_database = None
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)


class StoreUnavailable(RuntimeError):
    pass


class RejoinConfirmationRequired(RuntimeError):
    def __init__(self, session_title: str) -> None:
        super().__init__("This invite rejoins a grouped session you previously left.")
        self.session_title = session_title


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _db():
    global _client, _database
    if _database is not None:
        return _database
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise StoreUnavailable("MongoDB is not configured. Set MONGODB_URI in .env.")
    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _database = _client[os.getenv("MONGODB_DATABASE", "nexa")]
        _database.users.create_index("email", unique=True)
        _database.auth_sessions.create_index("token_hash", unique=True)
        _database.auth_sessions.create_index("expires_at", expireAfterSeconds=0)
        _database.chat_sessions.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        _database.chat_messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
        _database.chat_messages.create_index([("session_id", ASCENDING), ("role", ASCENDING), ("sender_user_id", ASCENDING), ("created_at", ASCENDING)])
        _database.chat_participants.create_index([("session_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
        _database.chat_participants.create_index([("user_id", ASCENDING), ("status", ASCENDING)])
        _database.chat_invites.create_index("token_hash", unique=True)
        _database.chat_invites.create_index("expires_at", expireAfterSeconds=0)
        return _database
    except PyMongoError as exc:
        _client = None
        _database = None
        raise StoreUnavailable("MongoDB could not be reached. Check MONGODB_URI and network access.") from exc


def _normalise_email(email: str) -> str:
    return email.strip().lower()


def user_public(user: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(user["id"]),
        "name": str(user["name"]),
        "email": str(user["email"]),
        "picture": str(user.get("picture") or ""),
    }


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}${digest.hex()}"


def _password_matches(password: str, saved: str) -> bool:
    try:
        salt_hex, digest_hex = saved.split("$", 1)
        candidate = _password_hash(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def create_password_user(name: str, email: str, password: str) -> dict[str, str]:
    email = _normalise_email(email)
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if not email or "@" not in email:
        raise ValueError("Enter a valid email address.")
    user = {
        "id": str(uuid.uuid4()),
        "name": " ".join(name.split()) or email.split("@", 1)[0],
        "email": email,
        "password_hash": _password_hash(password),
        "provider": "password",
        "created_at": _now(),
    }
    try:
        _db().users.insert_one(user)
    except PyMongoError as exc:
        if "duplicate key" in str(exc).lower():
            raise ValueError("An account already exists for this email. Sign in instead.") from exc
        raise
    return user_public(user)


def password_user(email: str, password: str) -> dict[str, str] | None:
    user = _db().users.find_one({"email": _normalise_email(email)})
    if not user or not _password_matches(password, str(user.get("password_hash") or "")):
        return None
    return user_public(user)


def google_user(name: str, email: str, picture: str = "") -> dict[str, str]:
    email = _normalise_email(email)
    if not email:
        raise ValueError("Google did not provide an email address.")
    db = _db()
    existing = db.users.find_one({"email": email})
    if existing:
        updates = {"provider": "google", "name": name or str(existing.get("name") or email.split("@", 1)[0])}
        if picture:
            updates["picture"] = picture
        db.users.update_one({"id": existing["id"]}, {"$set": updates})
        existing.update(updates)
        return user_public(existing)
    user = {
        "id": str(uuid.uuid4()),
        "name": name or email.split("@", 1)[0],
        "email": email,
        "picture": picture,
        "provider": "google",
        "created_at": _now(),
    }
    try:
        db.users.insert_one(user)
    except PyMongoError:
        existing = db.users.find_one({"email": email})
        if not existing:
            raise
        user = existing
    return user_public(user)


def create_auth_session(user: dict[str, str]) -> str:
    token = secrets.token_urlsafe(48)
    _db().auth_sessions.insert_one({
        "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "user_id": user["id"],
        "expires_at": _now() + timedelta(days=30),
    })
    return token


def auth_user(token: str) -> dict[str, str] | None:
    if not token:
        return None
    session = _db().auth_sessions.find_one({"token_hash": hashlib.sha256(token.encode()).hexdigest()})
    if not session:
        return None
    user = _db().users.find_one({"id": session["user_id"]})
    return user_public(user) if user else None


def revoke_auth_session(token: str) -> None:
    if token:
        _db().auth_sessions.delete_one({"token_hash": hashlib.sha256(token.encode()).hexdigest()})


def save_google_login_state(state: str) -> None:
    _db().google_login_states.insert_one({"state": state, "expires_at": _now() + timedelta(minutes=10)})


def consume_google_login_state(state: str) -> bool:
    return bool(_db().google_login_states.find_one_and_delete({"state": state}))


def create_chat_session(user_id: str, title: str = "New chat") -> dict[str, Any]:
    now = _now()
    item = {"id": str(uuid.uuid4()), "user_id": user_id, "title": title, "created_at": now, "updated_at": now}
    db = _db()
    db.chat_sessions.insert_one(item)
    db.chat_participants.insert_one({"session_id": item["id"], "user_id": user_id, "role": "admin", "status": "active", "joined_at": now, "last_read_at": now})
    return chat_session_public(item, user_id)


def chat_session_public(item: dict[str, Any], user_id: str = "") -> dict[str, Any]:
    db = _db()
    participant = db.chat_participants.find_one({"session_id": item["id"], "user_id": user_id, "status": "active"}) if user_id else None
    member_count = db.chat_participants.count_documents({"session_id": item["id"], "status": "active"}) if participant else 1
    unread_count = 0
    if participant and user_id:
        unread_query: dict[str, Any] = {
            "session_id": item["id"],
            "role": "user",
            "sender_user_id": {"$ne": user_id},
        }
        last_read_at = participant.get("last_read_at") or participant.get("joined_at")
        if last_read_at:
            unread_query["created_at"] = {"$gt": last_read_at}
        unread_count = db.chat_messages.count_documents(unread_query)
    return {
        "id": str(item["id"]),
        "title": str(item.get("title") or "New chat"),
        "created_at": item["created_at"].isoformat(),
        "updated_at": item["updated_at"].isoformat(),
        "role": str((participant or {}).get("role") or "admin"),
        "shared": member_count > 1,
        "member_count": member_count,
        "unread_count": unread_count,
        "has_unread": unread_count > 0,
        "private_copy": bool(item.get("private_copy")),
    }


def list_chat_sessions(user_id: str) -> list[dict[str, Any]]:
    db = _db()
    # Legacy sessions predate participants; claim their owner's membership lazily.
    for item in db.chat_sessions.find({"user_id": user_id}):
        db.chat_participants.update_one(
            {"session_id": item["id"], "user_id": user_id},
            {"$setOnInsert": {"role": "admin", "status": "active", "joined_at": item.get("created_at") or _now(), "last_read_at": item.get("created_at") or _now()}},
            upsert=True,
        )
    ids = [item["session_id"] for item in db.chat_participants.find({"user_id": user_id, "status": "active"}, {"session_id": 1})]
    return [chat_session_public(item, user_id) for item in db.chat_sessions.find({"id": {"$in": ids}}).sort("updated_at", DESCENDING)]


def owns_chat_session(user_id: str, session_id: str) -> bool:
    return bool(_db().chat_participants.find_one({"session_id": session_id, "user_id": user_id, "status": "active"}))


def session_participant(user_id: str, session_id: str) -> dict[str, Any] | None:
    return _db().chat_participants.find_one({"session_id": session_id, "user_id": user_id, "status": "active"})


def list_chat_participants(session_id: str) -> list[dict[str, Any]]:
    db = _db()
    participants = list(db.chat_participants.find({"session_id": session_id, "status": "active"}).sort("joined_at", ASCENDING))
    users = {item["id"]: item for item in db.users.find({"id": {"$in": [p["user_id"] for p in participants]}})}
    return [{"user_id": p["user_id"], "name": str(users.get(p["user_id"], {}).get("name") or "Member"), "email": str(users.get(p["user_id"], {}).get("email") or ""), "picture": str(users.get(p["user_id"], {}).get("picture") or ""), "role": p["role"], "joined_at": p["joined_at"].isoformat()} for p in participants]


def active_participant_count(session_id: str) -> int:
    return _db().chat_participants.count_documents({"session_id": session_id, "status": "active"})


def mark_chat_session_read(session_id: str, user_id: str) -> bool:
    result = _db().chat_participants.update_one(
        {"session_id": session_id, "user_id": user_id, "status": "active"},
        {"$set": {"last_read_at": _now()}},
    )
    return bool(result.matched_count)


def _invite_history_policy(history_mode: str) -> tuple[str, datetime | None]:
    if history_mode == "all":
        return "all", None
    if history_mode == "past_3_days":
        return "past_3_days", _now() - timedelta(days=3)
    return "none", _now()


def create_chat_invite(session_id: str, created_by: str, history_mode: str = "none") -> str:
    mode, history_since = _invite_history_policy(history_mode)
    token = secrets.token_urlsafe(32)
    _db().chat_invites.insert_one({
        "session_id": session_id,
        "created_by": created_by,
        "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "history_mode": mode,
        "history_since": history_since,
        "expires_at": _now() + timedelta(days=7),
    })
    return token


def _private_copy_for_source(db, user_id: str, source_session_id: str) -> dict[str, Any] | None:
    return db.chat_sessions.find_one(
        {"user_id": user_id, "private_copy": True, "source_session_id": source_session_id},
        sort=[("updated_at", DESCENDING)],
    )


def _merge_private_copy_messages(db, private_copy: dict[str, Any], target_session_id: str, user_id: str, share_private_conversation: bool, created_at: datetime) -> None:
    fork_created_at = private_copy.get("created_at") or created_at
    if isinstance(fork_created_at, datetime) and fork_created_at.tzinfo is None:
        fork_created_at = fork_created_at.replace(tzinfo=timezone.utc)
    copied_messages: list[dict[str, Any]] = []
    for message in db.chat_messages.find({
        "session_id": private_copy["id"],
        "created_at": {"$gte": fork_created_at},
        "role": {"$in": ["user", "assistant"]},
    }).sort("created_at", ASCENDING):
        copy = {key: value for key, value in message.items() if key != "_id"}
        copy.update({
            "id": str(uuid.uuid4()),
            "session_id": target_session_id,
            "created_at": created_at,
            "visibility": "shared" if share_private_conversation else "private",
            "visible_to_user_id": "" if share_private_conversation else user_id,
        })
        copied_messages.append(copy)
    if copied_messages:
        db.chat_messages.insert_many(copied_messages)
        db.chat_sessions.update_one({"id": target_session_id}, {"$set": {"updated_at": created_at}})
    db.chat_participants.update_one(
        {"session_id": private_copy["id"], "user_id": user_id},
        {"$set": {"status": "removed", "removed_at": created_at}},
    )


def accept_chat_invite(user_id: str, token: str, share_private_conversation: bool | None = None) -> dict[str, Any]:
    db = _db()
    invite = db.chat_invites.find_one({"token_hash": hashlib.sha256(token.encode()).hexdigest()})
    expires_at = invite.get("expires_at") if invite else None
    # PyMongo returns naive datetimes unless the client is configured with
    # tz_aware=True. Treat persisted naive values as UTC, matching _now().
    if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if not invite or not isinstance(expires_at, datetime) or expires_at <= _now():
        raise ValueError("This invitation is invalid or has expired.")
    session = db.chat_sessions.find_one({"id": invite["session_id"]})
    if not session:
        raise ValueError("This chat session is no longer available.")
    existing = db.chat_participants.find_one({"session_id": session["id"], "user_id": user_id, "status": "active"})
    if existing:
        return chat_session_public(session, user_id)
    removed_participant = db.chat_participants.find_one({"session_id": session["id"], "user_id": user_id, "status": "removed"})
    private_copy = _private_copy_for_source(db, user_id, session["id"]) if removed_participant else None
    if removed_participant and share_private_conversation is None:
        raise RejoinConfirmationRequired(str(session.get("title") or "this session"))
    joined_at = _now()
    history_mode = str(invite.get("history_mode") or "none")
    history_since = invite.get("history_since")
    if isinstance(history_since, datetime) and history_since.tzinfo is None:
        history_since = history_since.replace(tzinfo=timezone.utc)
    if history_mode not in {"all", "past_3_days", "none"}:
        history_mode = "none"
    if history_mode != "all" and not isinstance(history_since, datetime):
        history_since = joined_at
    db.chat_participants.update_one(
        {"session_id": session["id"], "user_id": user_id},
        {"$set": {
            "role": "member",
            "status": "active",
            "joined_at": joined_at,
            "last_read_at": joined_at,
            "history_mode": history_mode,
            "history_since": history_since,
            "history_shared_until": joined_at if history_mode in {"all", "past_3_days"} else None,
        }, "$unset": {"removed_at": ""}},
        upsert=True,
    )
    if private_copy:
        _merge_private_copy_messages(db, private_copy, session["id"], user_id, bool(share_private_conversation), joined_at)
    joined_user = db.users.find_one({"id": user_id}, {"email": 1}) or {}
    joined_email = str(joined_user.get("email") or "")
    save_message(
        session["id"],
        "system",
        f"{joined_email or 'A user'} is added",
        extra={"target_user_id": user_id, "target_email": joined_email, "system_action": "added"},
    )
    public = chat_session_public(session, user_id)
    public["invite_history_mode"] = history_mode
    public["rejoined"] = bool(private_copy)
    return public


def set_participant_role(session_id: str, user_id: str, role: str) -> bool:
    if role not in {"admin", "member"}:
        raise ValueError("Role must be admin or member.")
    if role == "member" and _db().chat_participants.count_documents({"session_id": session_id, "status": "active", "role": "admin"}) <= 1:
        raise ValueError("A shared chat must keep at least one admin.")
    return bool(_db().chat_participants.update_one({"session_id": session_id, "user_id": user_id, "status": "active"}, {"$set": {"role": role}}).matched_count)


def remove_participant_and_fork(session_id: str, user_id: str, notice_action: str = "removed") -> dict[str, Any]:
    db = _db()
    participant = session_participant(user_id, session_id)
    if not participant:
        raise ValueError("This user is not an active participant.")
    if participant["role"] == "admin" and db.chat_participants.count_documents({"session_id": session_id, "status": "active", "role": "admin"}) <= 1:
        raise ValueError("Assign another admin before removing the last admin.")
    source = db.chat_sessions.find_one({"id": session_id})
    if not source:
        raise ValueError("Chat session not found.")
    now = _now()
    fork = {"id": str(uuid.uuid4()), "user_id": user_id, "title": source.get("title") or "New chat", "private_copy": True, "source_session_id": session_id, "created_at": now, "updated_at": now}
    visible = {"$or": [{"visibility": {"$exists": False}}, {"visibility": "shared"}, {"visible_to_user_id": user_id}]}
    copies = []
    for message in db.chat_messages.find({"session_id": session_id, **visible}).sort("created_at", ASCENDING):
        copy = {key: value for key, value in message.items() if key != "_id"}
        copy.update({"id": str(uuid.uuid4()), "session_id": fork["id"]})
        copies.append(copy)
    db.chat_sessions.insert_one(fork)
    db.chat_participants.insert_one({"session_id": fork["id"], "user_id": user_id, "role": "admin", "status": "active", "joined_at": now})
    if copies:
        db.chat_messages.insert_many(copies)
    db.chat_participants.update_one({"session_id": session_id, "user_id": user_id}, {"$set": {"status": "removed", "removed_at": now}})
    removed_user = db.users.find_one({"id": user_id}, {"email": 1}) or {}
    removed_email = str(removed_user.get("email") or "")
    notice = "left the chat" if notice_action == "left" else "is removed"
    action = "left" if notice_action == "left" else "removed"
    save_message(
        session_id,
        "system",
        f"{removed_email or 'A user'} {notice}",
        extra={"target_user_id": user_id, "target_email": removed_email, "system_action": action},
    )
    if notice_action != "left":
        save_message(
            fork["id"],
            "system",
            "You were removed",
            visibility="private",
            visible_to_user_id=user_id,
            extra={"target_user_id": user_id, "target_email": removed_email, "system_action": "removed"},
        )
    return chat_session_public(fork, user_id)


def delete_chat_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    db = _db()
    session = db.chat_sessions.find_one({"id": session_id})
    if not session or not owns_chat_session(user_id, session_id):
        return None
    members = db.chat_participants.count_documents({"session_id": session_id, "status": "active"})
    if members > 1:
        return {"deleted": True, "private_session": remove_participant_and_fork(session_id, user_id, notice_action="left")}
    result = db.chat_sessions.delete_one({"id": session_id})
    if not result.deleted_count:
        return None
    db.chat_messages.delete_many({"session_id": session_id})
    return {"deleted": True, "session_id": session_id}


def _visible_message_query(session_id: str, user_id: str = "") -> dict[str, Any]:
    db = _db()
    query: dict[str, Any] = {"session_id": session_id}
    if user_id:
        visibility_filter = {"$or": [{"visibility": {"$exists": False}}, {"visibility": "shared"}, {"visible_to_user_id": user_id}]}
        query.update(visibility_filter)
        session = db.chat_sessions.find_one({"id": session_id}, {"private_copy": 1}) or {}
        participant = session_participant(user_id, session_id)
        if participant and not session.get("private_copy"):
            history_mode = str(participant.get("history_mode") or "none")
            history_since = participant.get("history_since") or participant["joined_at"]
            history_shared_until = participant.get("history_shared_until") or participant["joined_at"]
            if isinstance(history_since, datetime) and history_since.tzinfo is None:
                history_since = history_since.replace(tzinfo=timezone.utc)
            if isinstance(history_shared_until, datetime) and history_shared_until.tzinfo is None:
                history_shared_until = history_shared_until.replace(tzinfo=timezone.utc)
            query.pop("$or", None)
            if history_mode == "all":
                query["$or"] = [
                    visibility_filter,
                    {"created_at": {"$lte": history_shared_until}},
                ]
            elif history_mode == "past_3_days":
                query["$or"] = [
                    visibility_filter,
                    {"created_at": {"$gte": history_since, "$lte": history_shared_until}},
                ]
            else:
                query["$and"] = [
                    visibility_filter,
                    {"$or": [{"created_at": {"$gte": participant["joined_at"]}}, {"visible_to_user_id": user_id}]},
                ]
    return query


def _message_public(item: dict[str, Any]) -> dict[str, Any]:
    reply_to = item.get("reply_to")
    return {
        "id": str(item.get("id") or ""),
        "role": str(item["role"]),
        "content": str(item["content"]),
        "created_at": item["created_at"].isoformat(),
        "sender_name": str(item.get("sender_name") or ""),
        "sender_user_id": str(item.get("sender_user_id") or ""),
        "target_user_id": str(item.get("target_user_id") or ""),
        "target_email": str(item.get("target_email") or ""),
        "system_action": str(item.get("system_action") or ""),
        **({"reply_to": reply_to} if isinstance(reply_to, dict) else {}),
    }


def load_messages(session_id: str, limit: int = 200, user_id: str = "") -> list[dict[str, Any]]:
    db = _db()
    query = _visible_message_query(session_id, user_id)
    records = db.chat_messages.find(query).sort("created_at", ASCENDING).limit(limit)
    return [_message_public(item) for item in records]


def reply_snapshot(session_id: str, user_id: str, reply_to_id: str = "", max_chars: int = 280) -> dict[str, str] | None:
    if not reply_to_id:
        return None
    query = _visible_message_query(session_id, user_id)
    query["id"] = reply_to_id
    item = _db().chat_messages.find_one(query)
    if not item or item.get("role") == "system":
        return None
    content = str(item.get("content") or "")
    preview = " ".join(content.split())
    if max_chars > 0:
        content = preview[:max_chars].rstrip()
    else:
        content = content.strip()
    return {
        "id": str(item.get("id") or ""),
        "role": str(item.get("role") or ""),
        "content": content,
        "sender_name": str(item.get("sender_name") or ""),
        "sender_user_id": str(item.get("sender_user_id") or ""),
        "created_at": item["created_at"].isoformat(),
    }


def save_message(session_id: str, role: str, content: str, sender_user_id: str = "", sender_name: str = "", visibility: str = "shared", visible_to_user_id: str = "", reply_to: dict[str, str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    now = _now()
    item = {"id": str(uuid.uuid4()), "session_id": session_id, "role": role, "content": content, "sender_user_id": sender_user_id, "sender_name": sender_name, "visibility": visibility, "visible_to_user_id": visible_to_user_id, "created_at": now}
    if reply_to:
        item["reply_to"] = reply_to
    if extra:
        item.update(extra)
    db = _db()
    db.chat_messages.insert_one(item)
    title = " ".join(content.split())[:72] or "New chat"
    db.chat_sessions.update_one({"id": session_id}, {"$set": {"updated_at": now}})
    if role == "user":
        db.chat_sessions.update_one({"id": session_id, "title": "New chat"}, {"$set": {"title": title}})
    return _message_public(item)


def save_exchange(
    session_id: str,
    query: str,
    answer: str,
    answer_visibility: str = "shared",
    answer_visible_to_user_id: str = "",
    system_notice: str = "",
) -> None:
    db = _db()
    now = _now()
    latest_user = db.chat_messages.find_one({"session_id": session_id, "role": "user"}, sort=[("created_at", DESCENDING)])
    messages = []
    if not latest_user or str(latest_user.get("content") or "") != query:
        messages.append({"id": str(uuid.uuid4()), "session_id": session_id, "role": "user", "content": query, "visibility": "shared", "created_at": now})
    messages.append({
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": "assistant",
        "content": answer,
        "visibility": "private" if answer_visibility == "private" else "shared",
        "visible_to_user_id": answer_visible_to_user_id if answer_visibility == "private" else "",
        "created_at": now,
    })
    if system_notice and db.chat_participants.count_documents({"session_id": session_id, "status": "active"}) > 1:
        messages.append({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": "system",
            "content": system_notice,
            "visibility": "shared",
            "created_at": now,
        })
    db.chat_messages.insert_many(messages)
    title = " ".join(query.split())[:72] or "New chat"
    db.chat_sessions.update_one({"id": session_id}, {"$set": {"updated_at": now}, "$setOnInsert": {"title": title}})
    db.chat_sessions.update_one({"id": session_id, "title": "New chat"}, {"$set": {"title": title}})


class chat_session_context:
    def __init__(self, session_id: str, user_id: str = "") -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.token = None
        self.user_token = None

    def __enter__(self):
        self.token = _chat_session_id.set(self.session_id)
        self.user_token = _chat_user_id.set(self.user_id)
        return self

    def __exit__(self, *_: object) -> None:
        if self.token is not None:
            _chat_session_id.reset(self.token)
        if self.user_token is not None:
            _chat_user_id.reset(self.user_token)


def current_chat_session_id() -> str:
    return _chat_session_id.get()


def current_chat_user_id() -> str:
    return _chat_user_id.get()

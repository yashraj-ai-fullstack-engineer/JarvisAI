"""Centralized error handling for user-facing messages.

Maps technical exceptions to friendly, consistent responses.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCode:
    """Stable error codes for frontend handling."""
    UNKNOWN = "UNKNOWN_ERROR"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    PLANNING_FAILED = "PLANNING_FAILED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    EXTERNAL_SERVICE_DOWN = "EXTERNAL_SERVICE_DOWN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# User-friendly messages mapped from error codes
USER_MESSAGES = {
    ErrorCode.UNKNOWN: "Something went wrong. Let's try again in a moment.",
    ErrorCode.LLM_UNAVAILABLE: "I'm having trouble connecting to my language model. Please try again shortly.",
    ErrorCode.TOOL_EXECUTION_FAILED: "One of my tools ran into an issue. Let me try a different approach.",
    ErrorCode.PLANNING_FAILED: "I couldn't figure out how to handle that request. Could you rephrase it?",
    ErrorCode.SESSION_EXPIRED: "Your session has expired. Please refresh the page and sign in again.",
    ErrorCode.PERMISSION_DENIED: "I don't have permission to do that. Check your connected services.",
    ErrorCode.RATE_LIMITED: "Too many requests right now. Please wait a bit and try again.",
    ErrorCode.EXTERNAL_SERVICE_DOWN: "An external service I rely on is temporarily unavailable. Try again later.",
    ErrorCode.VALIDATION_ERROR: "That request doesn't look quite right. Please check and try again.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on my end. Let's reconnect in a moment.",
}


# Exception type → error code mapping
EXCEPTION_CODE_MAP: dict[type[BaseException], str] = {
    ConnectionError: ErrorCode.EXTERNAL_SERVICE_DOWN,
    TimeoutError: ErrorCode.EXTERNAL_SERVICE_DOWN,
    PermissionError: ErrorCode.PERMISSION_DENIED,
    ValueError: ErrorCode.VALIDATION_ERROR,
    KeyError: ErrorCode.INTERNAL_ERROR,
    AttributeError: ErrorCode.INTERNAL_ERROR,
    RuntimeError: ErrorCode.INTERNAL_ERROR,
}


def get_error_code(exc: BaseException) -> str:
    """Map an exception to a stable error code."""
    for exc_type, code in EXCEPTION_CODE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    # Check by name for common LLM/tool errors
    exc_name = type(exc).__name__.lower()
    if "llm" in exc_name or "localllm" in exc_name or "model" in exc_name:
        return ErrorCode.LLM_UNAVAILABLE
    if "tool" in exc_name:
        return ErrorCode.TOOL_EXECUTION_FAILED
    if "plan" in exc_name:
        return ErrorCode.PLANNING_FAILED
    if "rate" in exc_name or "quota" in exc_name or "429" in str(exc):
        return ErrorCode.RATE_LIMITED
    if "unauthorized" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
        return ErrorCode.PERMISSION_DENIED
    return ErrorCode.UNKNOWN


def friendly_message(exc: BaseException) -> str:
    """Return a user-friendly message for any exception."""
    code = get_error_code(exc)
    msg = USER_MESSAGES.get(code, USER_MESSAGES[ErrorCode.UNKNOWN])
    logger.warning("error.mapped code=%s type=%s", code, type(exc).__name__)
    return msg


def log_and_get_friendly(exc: BaseException, context: str = "") -> str:
    """Log the full exception (for debugging) and return friendly message."""
    logger.exception("error.context=%s type=%s", context, type(exc).__name__)
    return friendly_message(exc)


def format_error_event(exc: BaseException, context: str = "") -> dict[str, Any]:
    """Create a standardized error event for SSE/WS streams."""
    code = get_error_code(exc)
    return {
        "type": "error",
        "message": friendly_message(exc),
        "error_code": code,
        "context": context,
    }
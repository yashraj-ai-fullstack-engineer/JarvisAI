"""Selective LangSmith instrumentation for Nexa workflows.

Top-level trace metadata identifies the signed-in user and original query.
All span inputs and outputs remain masked so connected Gmail, Drive, and
Calendar data never leaves Nexa through LangSmith tracing.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, Mapping

from langsmith import Client, trace, tracing_context

from Backend.LLMProvider import get_config


logger = logging.getLogger("nexa.langsmith")
TRACE_SCHEMA_VERSION = "nexa-trace-v1"


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    """Tracing is explicit: an API key alone never enables external telemetry."""
    return _truthy(get_config("LANGSMITH_TRACING", "false")) and bool(
        get_config("LANGSMITH_API_KEY", "").strip()
    )


def trace_project() -> str:
    return get_config("LANGSMITH_PROJECT", "nexa").strip() or "nexa"


@lru_cache(maxsize=1)
def _client() -> Client | None:
    if not tracing_enabled():
        return None
    return Client(
        api_key=get_config("LANGSMITH_API_KEY", "").strip(),
        api_url=get_config("LANGSMITH_ENDPOINT", "").strip() or None,
        workspace_id=get_config("LANGSMITH_WORKSPACE_ID", "").strip() or None,
        hide_inputs=True,
        hide_outputs=True,
    )


def request_descriptor(value: str) -> dict[str, str]:
    """A masked placeholder for a raw query held in root trace metadata."""
    return {"query": str(value or "")}


def end_trace(run: Any, outputs: Mapping[str, Any] | None = None) -> None:
    """End a span; the LangSmith client masks all inputs and outputs."""
    if run is None:
        return
    try:
        run.end(outputs=dict(outputs or {}))
    except Exception as exc:  # pragma: no cover - network/client dependent
        logger.warning("langsmith.trace_end_failed error_type=%s", type(exc).__name__)


@contextmanager
def trace_operation(
    name: str,
    *,
    run_type: str = "chain",
    inputs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any | None]:
    """Create a best-effort span with parent propagation and masked I/O."""
    client = _client()
    if client is None:
        yield None
        return

    body_failed = False
    yielded_run = False
    try:
        with tracing_context(
            enabled=True,
            client=client,
            project_name=trace_project(),
            tags=["nexa", *(tags or [])],
            metadata={"trace_schema": TRACE_SCHEMA_VERSION, **dict(metadata or {})},
        ):
            with trace(
                name,
                run_type=run_type,  # type: ignore[arg-type]
                inputs=dict(inputs or {}),
                client=client,
                project_name=trace_project(),
                tags=["nexa", *(tags or [])],
                metadata={"trace_schema": TRACE_SCHEMA_VERSION, **dict(metadata or {})},
            ) as run:
                try:
                    yielded_run = True
                    yield run
                except BaseException:
                    body_failed = True
                    raise
    except BaseException as exc:
        if body_failed:
            raise
        # Telemetry unavailability is observable in local logs but never makes
        # a customer request fail.
        logger.warning("langsmith.trace_unavailable name=%s error_type=%s", name, type(exc).__name__)
        if not yielded_run:
            yield None

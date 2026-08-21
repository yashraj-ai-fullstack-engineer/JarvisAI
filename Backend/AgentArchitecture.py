"""Production primitives for Nexa's workflow-based agent runtime.

This module deliberately keeps routing and context construction deterministic.
An LLM may plan *inside* a selected workflow, but it never gets to grant itself
new permissions or bypass the workflow policy.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from pydantic import BaseModel, Field

from Backend.Chatbot import LoadHistory
from Backend.GoogleOAuth import google_mcp_connected
from Backend.MongoStore import current_chat_session_id, current_chat_user_id
from Backend.OwnerRAG import is_owner_question
from Backend.Paths import DATA_DIR
from Backend.SessionContext import load_for_agent, prompt_block


class Workflow(str, Enum):
    DIRECT = "direct"
    RESEARCH = "research"
    PERSONAL_APP = "personal_app"
    ACTION = "action"
    KNOWLEDGE = "knowledge"
    LONG_RUNNING = "long_running"


class RouteDecision(BaseModel):
    """Validated boundary between intake and agent execution."""

    workflow: Workflow
    domains: list[str] = Field(default_factory=list)
    reason: str
    requires_confirmation: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


class AgentContext(BaseModel):
    """Small, user-scoped context supplied to a workflow, never raw history."""

    conversation: str = ""
    session_context: dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""


class AgentRun(BaseModel):
    id: str
    started_at: str
    completed_at: str = ""
    status: str = "running"
    workflow: str
    domains: list[str] = Field(default_factory=list)
    session_id: str = ""
    user_id: str = ""
    request_chars: int = 0
    selected_tools: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    error: str = ""


_ACTION_PATTERN = re.compile(
    r"\b(?:send|draft|compose|schedule|create|update|delete|remove|cancel|reply|"
    r"book|open|launch|close|mute|unmute|set|increase|decrease|upload|share)\b",
    re.I,
)
_RESEARCH_PATTERN = re.compile(
    r"\b(?:latest|current|today|news|research|compare|price|stock|weather|"
    r"forecast|exchange rate|near me|directions?|find online|search(?: the)? web)\b",
    re.I,
)
_LONG_RUNNING_PATTERN = re.compile(
    r"\b(?:deep research|comprehensive research|monitor|every day|daily briefing|"
    r"scan .*repository|analyse .*repository|large report)\b",
    re.I,
)
_DEVICE_READ_PATTERN = re.compile(
    r"\b(?:battery|wi-?fi|system specs?|laptop specs?|processor|ram|gpu|"
    r"storage|windows version|power status|power and wifi status)\b",
    re.I,
)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def route_request(
    query: str,
    connected_plugin_domains: Iterable[str] = (),
    session_context: dict[str, Any] | None = None,
) -> RouteDecision:
    """Choose a workflow using explicit, auditable rules.

    Ambiguous requests intentionally go to the direct workflow. Its scoped
    planner can still request bounded web research where appropriate; this
    avoids confidently routing a vague request to a private connector.
    """
    text = _normalise(query)
    if _LONG_RUNNING_PATTERN.search(text):
        return RouteDecision(
            workflow=Workflow.LONG_RUNNING,
            domains=["jobs"],
            reason="The request describes work that can exceed an interactive run.",
            confidence=0.92,
        )

    domains: list[str] = []
    if re.search(r"\b(?:gmail|e-?mails?|inbox|unread|message from)\b", text):
        domains.append("gmail")
    if re.search(r"\b(?:google drive|my drive|drive file|google doc|google sheet|google slide)\b", text):
        domains.append("drive")
    if re.search(r"\b(?:calendars?|meetings?|appointments?|events?|availability|free time|busy)\b", text):
        domains.append("calendar")
    for domain in connected_plugin_domains:
        normalized_domain = _normalise(domain).replace(" ", "_")
        phrase = normalized_domain.replace("_", " ")
        if phrase and re.search(rf"\b{re.escape(phrase)}\b", text):
            domains.append(normalized_domain)
    if is_owner_question(text):
        return RouteDecision(
            workflow=Workflow.KNOWLEDGE,
            domains=["owner_profile"],
            reason="The request targets Nexa's private owner-profile knowledge.",
            confidence=0.93,
        )

    action = bool(_ACTION_PATTERN.search(text))
    if domains:
        return RouteDecision(
            workflow=Workflow.ACTION if action else Workflow.PERSONAL_APP,
            domains=domains,
            reason="The request explicitly names a connected personal application.",
            requires_confirmation=action,
            confidence=0.95,
        )
    if action and re.search(r"\b(?:app|application|volume|brightness|screen|document|file)\b", text):
        return RouteDecision(
            workflow=Workflow.ACTION,
            domains=["device"],
            reason="The request explicitly asks Nexa to change local or external state.",
            requires_confirmation=True,
            confidence=0.9,
        )
    if _DEVICE_READ_PATTERN.search(text):
        return RouteDecision(
            workflow=Workflow.DIRECT,
            domains=["device"],
            reason="The request asks for read-only information about this computer.",
            confidence=0.9,
        )
    if _RESEARCH_PATTERN.search(text):
        return RouteDecision(
            workflow=Workflow.RESEARCH,
            domains=["web"],
            reason="The request likely needs current or externally verified information.",
            confidence=0.82,
        )
    # Continue the previous workflow only for an explicitly referential
    # follow-up.  The session context is a routing hint; the planner still
    # receives the full bounded transcript and must validate the final plan.
    if session_context and re.search(
        r"\b(?:it|its|this|that|these|those|them|the first|the second|the last|"
        r"which one|what about|how about|compare|continue|again|also|earlier|previously)\b",
        text,
        re.I,
    ):
        prior_workflow = str(session_context.get("last_workflow") or "")
        try:
            prior = Workflow(prior_workflow)
        except ValueError:
            prior = Workflow.DIRECT
        if prior == Workflow.RESEARCH:
            return RouteDecision(
                workflow=Workflow.RESEARCH,
                domains=list(session_context.get("last_domains") or ["web"]),
                reason="The request is an ambiguous follow-up to the session's research workflow.",
                confidence=0.68,
            )
        if prior == Workflow.PERSONAL_APP:
            domains = list(session_context.get("last_domains") or [])
            if domains:
                return RouteDecision(
                    workflow=Workflow.PERSONAL_APP,
                    domains=domains,
                    reason="The request continues the session's connected-app conversation.",
                    confidence=0.68,
                )
        if prior == Workflow.KNOWLEDGE:
            return RouteDecision(
                workflow=Workflow.KNOWLEDGE,
                domains=list(session_context.get("last_domains") or ["owner_profile"]),
                reason="The request continues the session's private knowledge conversation.",
                confidence=0.68,
            )
    return RouteDecision(
        workflow=Workflow.DIRECT,
        reason="The request can be answered without selecting a private or mutating domain.",
        confidence=0.7,
    )


def build_context(max_messages: int = 10, max_chars: int = 3_000) -> AgentContext:
    """Return a bounded history view from the current, already user-scoped chat."""
    session_context = load_for_agent()
    if session_context:
        conversation = prompt_block(session_context)
        if max_chars > 0:
            conversation = conversation[:max_chars]
        return AgentContext(
            conversation=conversation,
            session_context=session_context,
            session_id=current_chat_session_id(),
            user_id=current_chat_user_id(),
        )
    try:
        messages = LoadHistory(limit=max_messages)
    except Exception:
        messages = []
    snippets: list[str] = []
    used = 0
    for message in reversed(messages):
        role = str(message.get("role") or "user")
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(message.get("content") or "").split())
        if not content:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = content[:remaining]
        snippets.append(f"{role.title()}: {content}")
        used += len(content)
    snippets.reverse()
    return AgentContext(
        conversation="\n".join(snippets),
        session_context={},
        session_id=current_chat_session_id(),
        user_id=current_chat_user_id(),
    )


def connected_plugin_domains(tools: Iterable[Any]) -> set[str]:
    """Discover active MCP plugin prefixes without hard-coding integrations."""
    built_in_prefixes = {
        "gmail", "google", "maps", "get", "check", "convert", "draft",
        "send", "create", "open", "close", "control", "read", "search",
        "research", "answer",
    }
    domains: set[str] = set()
    for tool in tools:
        name = str(getattr(tool, "name", "") or "")
        if "_" not in name:
            continue
        prefix = name.split("_", 1)[0].strip().lower()
        if prefix and prefix not in built_in_prefixes:
            domains.add(prefix)
    return domains


def tools_for_workflow(decision: RouteDecision, tools: Iterable[Any]) -> list[Any]:
    """Apply a deny-by-default tool boundary before the planner sees tools."""
    tools_by_name = {
        str(getattr(tool, "name", "") or ""): tool for tool in tools
        if str(getattr(tool, "name", "") or "")
    }
    names: set[str] = {"get_capabilities"}
    google_domains = {
        "gmail": "gmail",
        "drive": "google_drive",
        "calendar": "google_calendar",
    }

    def connected(domain: str) -> bool:
        service = google_domains.get(domain)
        return not service or google_mcp_connected(service)

    def connector_tools(prefix: str, *, allow_mutations: bool) -> set[str]:
        mutation_markers = (
            "create", "update", "delete", "remove", "send", "reply", "respond",
            "archive", "trash", "upload", "move", "copy", "share", "label",
        )
        names = {name for name in tools_by_name if name.startswith(prefix)}
        if allow_mutations:
            return names
        return {
            name for name in names
            if not any(marker in name.removeprefix(prefix).lower().split("_") for marker in mutation_markers)
        }
    if decision.workflow in {Workflow.DIRECT, Workflow.RESEARCH, Workflow.LONG_RUNNING}:
        names.update({
            "research_web", "search_web", "read_webpage", "open_website",
            "maps_search_places", "maps_geocode", "maps_get_directions",
            "get_weather_and_air_quality", "check_holiday_schedule", "convert_currency",
        })
    elif decision.workflow == Workflow.KNOWLEDGE:
        names.add("answer_owner_profile")
    elif decision.workflow == Workflow.PERSONAL_APP:
        for domain in decision.domains:
            if not connected(domain):
                continue
            prefixes = {"drive": "google_drive_", "calendar": "google_calendar_"}
            prefix = prefixes.get(domain, f"{domain}_")
            names.update(connector_tools(prefix, allow_mutations=False))
    elif decision.workflow == Workflow.ACTION:
        names.update({
            "draft_email", "create_document", "open_application",
            "close_application", "control_volume", "control_brightness",
            "get_system_specs", "get_power_and_wifi_status",
        })
        if google_mcp_connected("gmail"):
            names.add("send_email")
        for domain in decision.domains:
            if not connected(domain):
                continue
            prefixes = {"drive": "google_drive_", "calendar": "google_calendar_"}
            prefix = prefixes.get(domain, f"{domain}_")
            names.update(name for name in tools_by_name if name.startswith(prefix))
    if "device" in decision.domains:
        names.update({"get_system_specs", "get_power_and_wifi_status"})
    return [tool for name, tool in tools_by_name.items() if name in names]


class AgentRunStore:
    """Append-only, redacted runtime audit records for troubleshooting and evals."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "AgentRuns.json"
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, values: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(values[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def start(self, decision: RouteDecision, request: str, selected_tools: Iterable[str]) -> AgentRun:
        run = AgentRun(
            id=str(uuid4()),
            started_at=datetime.now(timezone.utc).isoformat(),
            workflow=decision.workflow.value,
            domains=decision.domains,
            session_id=current_chat_session_id(),
            user_id=current_chat_user_id(),
            request_chars=len(request),
            selected_tools=sorted(set(selected_tools)),
        )
        with self._lock:
            values = self._read()
            values.append(run.model_dump())
            self._write(values)
        return run

    def finish(self, run_id: str, *, tool_calls: Iterable[str] = (), error: str = "") -> None:
        with self._lock:
            values = self._read()
            for item in reversed(values):
                if item.get("id") == run_id:
                    item.update({
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "failed" if error else "completed",
                        "tool_calls": list(tool_calls),
                        "error": error[:500],
                    })
                    break
            self._write(values)

    def set_selected_tools(self, run_id: str, selected_tools: Iterable[str]) -> None:
        """Record the validated plan only after it has passed policy checks."""
        with self._lock:
            values = self._read()
            for item in reversed(values):
                if item.get("id") == run_id:
                    item["selected_tools"] = sorted(set(selected_tools))
                    break
            self._write(values)

    def list_for_current_user(self, limit: int = 50) -> list[dict[str, Any]]:
        user_id = current_chat_user_id()
        with self._lock:
            records = self._read()
        if user_id:
            records = [item for item in records if item.get("user_id") == user_id]
        return list(reversed(records[-max(1, min(limit, 100)):]))


RUN_STORE = AgentRunStore()

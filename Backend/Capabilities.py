"""Single source of truth for Nexa capabilities, policies, and tool routing."""

from __future__ import annotations

import re
from typing import Any, Iterable

from Backend.GoogleOAuth import google_mcp_connected
from Backend.MapsConnector import geoapify_configured


LOCAL_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "id": "general",
        "label": "General assistant",
        "description": "Answer normal questions and continue conversations.",
        "category": "knowledge",
        "tools": [],
        "side_effect": "none",
    },
    {
        "id": "web",
        "label": "Live web research",
        "description": "Search the public web, inspect sources, and cite URLs.",
        "category": "knowledge",
        "tools": ["search_web", "read_webpage", "open_website"],
        "side_effect": "open_website_only",
    },
    {
        "id": "maps",
        "label": "Places and directions",
        "description": "Find nearby places, resolve locations, and calculate driving, walking, or bicycle routes through Geoapify.",
        "category": "location",
        "tools": ["maps_search_places", "maps_geocode", "maps_get_directions"],
        "side_effect": "none",
    },
    {
        "id": "live_planning",
        "label": "Weather, holidays, and currency",
        "description": "Check live weather and air quality, evaluate public holidays for scheduling, and convert currencies using reference rates.",
        "category": "live_data",
        "tools": ["get_weather_and_air_quality", "check_holiday_schedule", "convert_currency"],
        "side_effect": "none",
    },
    {
        "id": "device",
        "label": "Windows device control",
        "description": "Open or close applications, control audio and brightness, and inspect this computer.",
        "category": "local_action",
        "tools": [
            "open_application",
            "close_application",
            "control_volume",
            "control_brightness",
            "get_system_specs",
            "get_power_and_wifi_status",
        ],
        "side_effect": "explicit_request",
    },
    {
        "id": "email",
        "label": "Email composition and sending",
        "description": "Draft locally and send from the connected Gmail account only after UI confirmation.",
        "category": "connected_action",
        "tools": ["draft_email", "send_email"],
        "side_effect": "confirmation_required_to_send",
    },
    {
        "id": "documents",
        "label": "Local documents",
        "description": "Write requested content to a local text document.",
        "category": "local_action",
        "tools": ["create_document"],
        "side_effect": "explicit_request",
    },
    {
        "id": "owner_profile",
        "label": "Owner resume knowledge",
        "description": "Answer supported owner-profile questions from the local resume index.",
        "category": "private_knowledge",
        "tools": ["answer_owner_profile"],
        "side_effect": "none",
    },
)


GOOGLE_CAPABILITIES: dict[str, dict[str, Any]] = {
    "gmail": {
        "label": "Gmail",
        "description": "Search and read Gmail. Email sending uses this connected account and always requires UI confirmation.",
        "read_only_mcp": True,
        "features": [
            "search inbox threads",
            "read messages and threads",
            "list drafts and labels",
            "send after confirmation",
        ],
    },
    "google_drive": {
        "label": "Google Drive",
        "description": "Search, inspect, download, and read eligible Drive files without changing Drive.",
        "read_only_mcp": True,
        "features": [
            "search files",
            "list recent files",
            "read file content",
            "inspect metadata and permissions",
            "download file content",
        ],
    },
    "google_calendar": {
        "label": "Google Calendar",
        "description": "Read schedules and manage events. Every event mutation requires UI confirmation.",
        "read_only_mcp": False,
        "features": [
            "list and search events",
            "inspect calendars and events",
            "suggest meeting times",
            "create, update, delete, or respond after confirmation",
        ],
    },
}


_CAPABILITY_QUERY = re.compile(
    r"\b(?:what can you do|your capabilities|available tools|which tools|help me with|connected apps?)\b",
    re.I,
)
_EMAIL_QUERY = re.compile(
    (
        r"\b(?:gmail|emails?|e-mails?|mails?|inbox|draft|recipient|subject line|cc|bcc|"
        r"unread messages?|sent me|message from)\b"
        r"|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
    ),
    re.I,
)
_DRIVE_QUERY = re.compile(
    (
        r"\b(?:google drive|my drive|drive file|shared file|cloud file|spreadsheet|"
        r"google doc|google sheet|google slide)\b"
        r"|\b(?:find|search|read|open|summarize)\b.*\b(?:my )?(?:file|document|pdf)\b"
    ),
    re.I,
)
_CALENDAR_QUERY = re.compile(
    (
        r"\b(?:google calendar|calendar|schedule|meeting|appointment|event|"
        r"availability|free time|busy|invitation|invite)\b"
        r"|\b(?:am i free|what do i have|anything planned)\b"
    ),
    re.I,
)
_MAPS_QUERY = re.compile(
    (
        r"\b(?:near me|from me|nearby|closest|nearest|directions?|route|how (?:do|can) i get|"
        r"how far|how long|distance|travel time|away|restaurants?|cafes?|coffee shops?|hotels?|attractions?|"
        r"pharmacies|hospitals?|gas stations?|petrol pumps?|atms?|parking)\b"
    ),
    re.I,
)
_WEATHER_QUERY = re.compile(
    r"\b(?:weather|forecast|temperature|rain|raining|windy|humidity|air quality|aqi|pollution|pm2(?:\.5)?|smog|uv index)\b",
    re.I,
)
_HOLIDAY_QUERY = re.compile(
    r"\b(?:holiday|public holiday|bank holiday|working day|business day|workday|weekend|schedule.*(?:around|after|before)|avoid.*holiday)\b",
    re.I,
)
_CURRENCY_QUERY = re.compile(
    r"\b(?:convert|conversion|exchange rate|currency|forex|usd|inr|eur|gbp|jpy|cad|aud)\b",
    re.I,
)
_OWNER_QUERY = re.compile(
    (
        r"\b(?:yashraj|your owner|your creator|your developer|"
        r"your master|who is (?:the )?master|"
        r"who (?:made|built|created|developed) you|"
        r"owner (?:profile|resume|skills|projects|education|experience))\b"
    ),
    re.I,
)
_DOCUMENT_QUERY = re.compile(
    r"\b(?:write|create|make|prepare|draft)\b.*\b(?:document|letter|application|note|text file)\b",
    re.I,
)
_OPEN_WEB_QUERY = re.compile(
    r"\b(?:open|visit|navigate|browse|play|go to|take me to)\b",
    re.I,
)
_DEVICE_QUERY = re.compile(
    (
        r"\b(?:close|quit|mute|unmute|volume|brightness|dim|brighten|battery|"
        r"wi-?fi|system specs?|laptop specs?|processor|ram|gpu|storage|windows version)\b"
        r"|\b(?:open|launch|start|run)\b.*\b(?:app|application|program|notepad|"
        r"calculator|terminal|powershell|command prompt|vs code|visual studio|"
        r"file explorer|settings)\b"
    ),
    re.I,
)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def select_tools_for_query(query: str, tools: Iterable[Any]) -> list[Any]:
    """Expose a small, relevant toolset to the model for this request."""
    available = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    selected: set[str] = {"get_capabilities"}
    normalized = " ".join(query.split())

    # Web research remains available for unfamiliar or changing facts. Keeping
    # these two tools in the base set avoids relying on a brittle freshness
    # keyword classifier.
    selected.update({"search_web", "read_webpage"})

    # Connected Gmail read tools are always visible to the first brain. The
    # model decides whether a request is actually about email; keyword routing
    # must never be the gate that hides an available personal-data capability.
    selected.update(name for name in available if name.startswith("gmail_"))

    if _CAPABILITY_QUERY.search(normalized):
        return [available[name] for name in available if name in selected]

    if _OPEN_WEB_QUERY.search(normalized):
        selected.add("open_website")

    if _EMAIL_QUERY.search(normalized):
        selected.update({"draft_email", "send_email"})
        selected.update(name for name in available if name.startswith("gmail_"))

    if _DRIVE_QUERY.search(normalized):
        selected.update(name for name in available if name.startswith("google_drive_"))

    if _CALENDAR_QUERY.search(normalized):
        selected.update(name for name in available if name.startswith("google_calendar_"))

    if _MAPS_QUERY.search(normalized):
        selected.update(name for name in available if name.startswith("maps_"))

    if _WEATHER_QUERY.search(normalized):
        selected.add("get_weather_and_air_quality")

    if _HOLIDAY_QUERY.search(normalized):
        selected.add("check_holiday_schedule")

    if _CURRENCY_QUERY.search(normalized):
        selected.add("convert_currency")

    if _OWNER_QUERY.search(normalized):
        selected.add("answer_owner_profile")

    if _DOCUMENT_QUERY.search(normalized) and not _EMAIL_QUERY.search(normalized):
        selected.add("create_document")

    if _DEVICE_QUERY.search(normalized):
        selected.update({
            "open_application",
            "close_application",
            "control_volume",
            "control_brightness",
            "get_system_specs",
            "get_power_and_wifi_status",
        })

    # Third-party MCP servers remain discoverable by their prefixed server name.
    google_prefixes = ("gmail_", "google_drive_", "google_calendar_")
    local_tool_names = {
        str(tool_name)
        for capability in LOCAL_CAPABILITIES
        for tool_name in capability.get("tools", [])
    } | {"get_capabilities"}
    for name in available:
        if (
            name in local_tool_names
            or name.startswith(google_prefixes)
            or "_" not in name
        ):
            continue
        server_hint = name.split("_", 1)[0].replace("_", " ")
        if server_hint and server_hint in normalized.lower():
            selected.add(name)

    return [tool for name, tool in available.items() if name in selected]


def capability_snapshot(mcp_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build public, machine-readable live capability state."""
    mcp_snapshot = mcp_snapshot or {}
    server_by_name = {
        str(server.get("name")): server
        for server in mcp_snapshot.get("servers", [])
        if isinstance(server, dict)
    }
    google: list[dict[str, Any]] = []
    for service, definition in GOOGLE_CAPABILITIES.items():
        server = server_by_name.get(service, {})
        connector_connected = google_mcp_connected(service)
        google.append({
            "id": service,
            "label": definition["label"],
            "description": definition["description"],
            "features": list(definition["features"]),
            "connected": connector_connected,
            "available": connector_connected,
            "read_only": bool(definition["read_only_mcp"]),
            "error": str(server.get("runtime_error") or ""),
        })
    local = []
    for item in LOCAL_CAPABILITIES:
        available = geoapify_configured() if item["id"] == "maps" else True
        local.append(dict(item, available=available))
    return {
        "local": local,
        "google": google,
        "plugins": [
            {
                "id": str(server.get("name") or ""),
                "label": str(server.get("label") or server.get("name") or "Connected app"),
                "description": str(server.get("description") or ""),
                "connected": bool(server.get("oauth_connected", True)),
                "available": bool(server.get("ready")),
                "read_only": bool(server.get("read_only")),
            }
            for server in mcp_snapshot.get("servers", [])
            if isinstance(server, dict) and str(server.get("name") or "") not in GOOGLE_CAPABILITIES
        ],
        "mcp_runtime": dict(mcp_snapshot.get("runtime") or {}),
    }


def capability_prompt(
    mcp_snapshot: dict[str, Any] | None,
    exposed_tool_names: Iterable[str],
) -> str:
    """Generate concise capability context for the system prompt."""
    snapshot = capability_snapshot(mcp_snapshot)
    tool_names = sorted({name for name in exposed_tool_names if name})
    lines = [
        "Live capability state:",
        "- Normal conversation and live public-web research are available.",
        "- Local Windows actions, local documents, and owner-resume retrieval are available when relevant.",
    ]
    maps = next((item for item in snapshot["local"] if item["id"] == "maps"), None)
    if maps:
        state = "configured and ready" if maps["available"] else "not configured"
        lines.append(f"- Geoapify places and directions: {state}. {maps['description']}")
    lines.append("- Live planning data: weather and AQI through Open-Meteo, public-holiday scheduling through Nager, and reference currency conversion through Frankfurter.")
    for service in snapshot["google"]:
        if service["available"]:
            state = "connected and ready"
        elif service["connected"]:
            state = "connected but currently unavailable"
        else:
            state = "not connected"
        lines.append(f"- {service['label']}: {state}. {service['description']}")
    for plugin in snapshot["plugins"]:
        state = "connected and ready" if plugin["available"] else "not connected or unavailable"
        lines.append(
            f"- Connected plugin {plugin['label']}: {state}. {plugin['description']}"
        )
    if tool_names:
        lines.append(f"- Tools exposed for this request: {', '.join(tool_names)}.")
    lines.append(
        "- Never claim an unavailable or disconnected capability worked. Explain the prerequisite instead."
    )
    return "\n".join(lines)

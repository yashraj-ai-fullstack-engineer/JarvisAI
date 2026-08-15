from langchain_core.messages import ToolMessage

from Backend.AgentTools import AGENT_TOOLS
from Backend.JarvisAgent import _normalize_web_plan, _web_tool_limit_reached


def test_planner_tool_selection_is_not_rewritten():
    plan = {
        "tool_names": ["search_web", "read_webpage"],
        "workflow": ["Search repeatedly", "Read pages"],
        "max_tool_calls": 4,
    }

    normalized = _normalize_web_plan(plan, AGENT_TOOLS)

    assert normalized == plan


def test_explicit_open_website_flow_keeps_search_web():
    plan = {
        "tool_names": ["search_web", "open_website"],
        "workflow": ["Find official site", "Open it"],
        "max_tool_calls": 2,
    }

    normalized = _normalize_web_plan(plan, AGENT_TOOLS)

    assert normalized == plan


def test_research_web_cannot_repeat_after_success():
    messages = [
        ToolMessage(content='{"ok": true}', name="research_web", tool_call_id="research-1"),
    ]
    next_calls = [{"name": "research_web"}]

    assert _web_tool_limit_reached(messages, next_calls)


def test_search_web_has_three_call_cap():
    messages = [
        ToolMessage(content='{"ok": true}', name="search_web", tool_call_id=f"search-{index}")
        for index in range(3)
    ]
    next_calls = [{"name": "search_web"}]

    assert _web_tool_limit_reached(messages, next_calls)

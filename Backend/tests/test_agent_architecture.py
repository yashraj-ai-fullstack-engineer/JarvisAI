from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from Backend.AgentArchitecture import (
    AgentRunStore,
    RouteDecision,
    Workflow,
    build_context,
    route_request,
    tools_for_workflow,
)


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


def names(decision: RouteDecision) -> set[str]:
    tools = [
        FakeTool("get_capabilities"),
        FakeTool("research_web"),
        FakeTool("gmail_search_messages"),
        FakeTool("google_drive_search_files"),
        FakeTool("google_calendar_list_events"),
        FakeTool("draft_email"),
        FakeTool("send_email"),
        FakeTool("answer_owner_profile"),
    ]
    with patch("Backend.AgentArchitecture.google_mcp_connected", return_value=True):
        return {tool.name for tool in tools_for_workflow(decision, tools)}


class AgentArchitectureTests(TestCase):
    def test_email_read_routes_to_personal_gmail_only(self) -> None:
        decision = route_request("Summarize my unread Gmail messages")
        self.assertEqual(decision.workflow, Workflow.PERSONAL_APP)
        self.assertEqual(decision.domains, ["gmail"])
        self.assertEqual(names(decision), {"get_capabilities", "gmail_search_messages"})

    def test_send_email_routes_to_action_and_keeps_confirmation_flag(self) -> None:
        decision = route_request("Send an email to alex@example.com about the launch")
        self.assertEqual(decision.workflow, Workflow.ACTION)
        self.assertTrue(decision.requires_confirmation)
        allowed = names(decision)
        self.assertTrue({"gmail_search_messages", "draft_email", "send_email"} <= allowed)
        self.assertNotIn("google_drive_search_files", allowed)

    def test_drive_and_calendar_mixed_question_keeps_both_personal_domains(self) -> None:
        decision = route_request("Read my project plan from Google Drive and check my calendar")
        self.assertEqual(decision.workflow, Workflow.PERSONAL_APP)
        self.assertEqual(decision.domains, ["drive", "calendar"])
        self.assertTrue({"google_drive_search_files", "google_calendar_list_events"} <= names(decision))

    def test_owner_profile_isolated_from_web_and_personal_tools(self) -> None:
        decision = route_request("What does your creator's resume say about projects?")
        self.assertEqual(decision.workflow, Workflow.KNOWLEDGE)
        self.assertEqual(names(decision), {"get_capabilities", "answer_owner_profile"})

    def test_disconnected_gmail_is_not_exposed_to_the_planner(self) -> None:
        decision = route_request("Summarize my latest emails")
        self.assertEqual(decision.workflow, Workflow.PERSONAL_APP)
        self.assertEqual(decision.domains, ["gmail"])
        tools = [FakeTool("get_capabilities"), FakeTool("gmail_search_messages"), FakeTool("send_email")]
        with patch("Backend.AgentArchitecture.google_mcp_connected", return_value=False):
            allowed = {tool.name for tool in tools_for_workflow(decision, tools)}
        self.assertEqual(allowed, {"get_capabilities"})

    def test_context_is_bounded_and_uses_user_scoped_history(self) -> None:
        with patch(
            "Backend.AgentArchitecture.LoadHistory",
            return_value=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "system", "content": "not included"},
            ],
        ):
            context = build_context(max_chars=100)
        self.assertIn("User: first question", context.conversation)
        self.assertIn("Assistant: first answer", context.conversation)
        self.assertNotIn("not included", context.conversation)

    def test_run_store_records_no_request_or_tool_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentRunStore(Path(temp_dir) / "runs.json")
            decision = RouteDecision(workflow=Workflow.RESEARCH, reason="test", confidence=1)
            run = store.start(decision, "secret user request", ["research_web"])
            store.finish(run.id, tool_calls=["research_web"])
            payload = json.loads((Path(temp_dir) / "runs.json").read_text(encoding="utf-8"))
        self.assertEqual(payload[0]["status"], "completed")
        self.assertEqual(payload[0]["request_chars"], len("secret user request"))
        self.assertNotIn("secret user request", json.dumps(payload))

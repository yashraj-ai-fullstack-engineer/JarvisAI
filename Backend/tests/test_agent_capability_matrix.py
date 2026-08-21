"""Acceptance tests for the capabilities exposed by the real agent boundary.

These tests do not call the internet, Gmail, Windows, or an LLM. They verify
that a request is routed to the correct workflow and that only the tools that
workflow is allowed to see are exposed to the planner.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from Backend.AgentArchitecture import Workflow, route_request, tools_for_workflow
from Backend.AgentTools import AGENT_TOOLS
from Backend.Capabilities import LOCAL_CAPABILITIES


@dataclass
class FakeTool:
    name: str


TOOL_NAMES = {
    "get_capabilities",
    "research_web",
    "search_web",
    "read_webpage",
    "open_website",
    "maps_search_places",
    "maps_geocode",
    "maps_get_directions",
    "get_weather_and_air_quality",
    "check_holiday_schedule",
    "convert_currency",
    "get_system_specs",
    "get_power_and_wifi_status",
    "open_application",
    "close_application",
    "control_volume",
    "control_brightness",
    "create_document",
    "draft_email",
    "send_email",
    "answer_owner_profile",
    "gmail_search_messages",
    "gmail_read_message",
    "gmail_send_message",
    "google_drive_search_files",
    "google_drive_read_file_content",
    "google_drive_create_file",
    "google_calendar_list_events",
    "google_calendar_create_event",
}


def fake_tools() -> list[FakeTool]:
    return [FakeTool(name) for name in sorted(TOOL_NAMES)]


class AgentCapabilityMatrixTests(unittest.TestCase):
    def allowed(self, request: str) -> set[str]:
        with patch("Backend.AgentArchitecture.google_mcp_connected", return_value=True):
            decision = route_request(request)
            return {
                tool.name
                for tool in tools_for_workflow(decision, fake_tools())
            }

    def test_all_declared_local_capability_tools_are_registered(self) -> None:
        registered = {str(getattr(tool, "name", "")) for tool in AGENT_TOOLS}
        missing: dict[str, list[str]] = {}
        for capability in LOCAL_CAPABILITIES:
            absent = [name for name in capability["tools"] if name not in registered]
            if absent:
                missing[str(capability["id"])] = absent
        self.assertEqual(missing, {}, f"Declared capabilities missing tools: {missing}")

    def test_general_question_cannot_see_private_or_mutating_tools(self) -> None:
        allowed = self.allowed("Explain recursion in simple language")
        self.assertIn("research_web", allowed)
        self.assertIn("get_capabilities", allowed)
        self.assertNotIn("gmail_search_messages", allowed)
        self.assertNotIn("google_drive_search_files", allowed)
        self.assertNotIn("send_email", allowed)
        self.assertNotIn("open_application", allowed)
        self.assertNotIn("get_system_specs", allowed)

    def test_live_planning_tools_are_available_for_the_main_agent_path(self) -> None:
        allowed = self.allowed("What is the weather and air quality in Delhi today?")
        self.assertIn("get_weather_and_air_quality", allowed)
        self.assertIn("maps_geocode", allowed)
        self.assertIn("check_holiday_schedule", allowed)
        self.assertIn("convert_currency", allowed)

    def test_maps_requests_receive_only_non_mutating_location_tools(self) -> None:
        allowed = self.allowed("Find restaurants near me")
        self.assertIn("maps_search_places", allowed)
        self.assertIn("maps_get_directions", allowed)
        self.assertNotIn("send_email", allowed)
        self.assertNotIn("open_application", allowed)

    def test_device_read_requests_receive_system_inspection_tools(self) -> None:
        allowed = self.allowed("What are my laptop specs and battery status?")
        self.assertIn("get_system_specs", allowed)
        self.assertIn("get_power_and_wifi_status", allowed)

    def test_device_mutations_receive_action_tools(self) -> None:
        allowed = self.allowed("Open Notepad and set brightness to 60 percent")
        self.assertIn("open_application", allowed)
        self.assertIn("control_brightness", allowed)
        self.assertNotIn("gmail_search_messages", allowed)

    def test_owner_questions_are_isolated_to_owner_profile_tool(self) -> None:
        with patch("Backend.AgentArchitecture.google_mcp_connected", return_value=True):
            decision = route_request("What projects are listed in your creator's resume?")
            allowed = {
                tool.name for tool in tools_for_workflow(decision, fake_tools())
            }
        self.assertEqual(decision.workflow, Workflow.KNOWLEDGE)
        self.assertEqual(allowed, {"get_capabilities", "answer_owner_profile"})

    def test_gmail_read_workflow_cannot_see_connector_mutations(self) -> None:
        allowed = self.allowed("Summarize my latest Gmail messages")
        self.assertIn("gmail_search_messages", allowed)
        self.assertIn("gmail_read_message", allowed)
        self.assertNotIn("gmail_send_message", allowed)

    def test_drive_read_workflow_is_strictly_read_only(self) -> None:
        allowed = self.allowed("Find my project plan in Google Drive")
        self.assertIn("google_drive_search_files", allowed)
        self.assertIn("google_drive_read_file_content", allowed)
        self.assertNotIn("google_drive_create_file", allowed)

    def test_calendar_mutation_is_not_available_for_a_read_request(self) -> None:
        allowed = self.allowed("What meetings do I have tomorrow?")
        self.assertIn("google_calendar_list_events", allowed)
        self.assertNotIn("google_calendar_create_event", allowed)

    def test_explicit_email_send_keeps_confirmation_capability(self) -> None:
        with patch("Backend.AgentArchitecture.google_mcp_connected", return_value=True):
            decision = route_request("Send an email to alex@example.com about the launch")
            self.assertEqual(decision.workflow, Workflow.ACTION)
            self.assertTrue(decision.requires_confirmation)
            allowed = {
                tool.name for tool in tools_for_workflow(decision, fake_tools())
            }
        self.assertIn("draft_email", allowed)
        self.assertIn("send_email", allowed)


if __name__ == "__main__":
    unittest.main()

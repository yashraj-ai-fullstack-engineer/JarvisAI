from __future__ import annotations

import unittest
from unittest.mock import patch

from Backend.AgentArchitecture import Workflow, build_context, route_request
from Backend.SessionContext import derive_context, follow_up_query, prompt_block


class SessionContextTests(unittest.TestCase):
    def test_shared_speakers_are_preserved_in_context(self) -> None:
        context = derive_context(
            [
                {"id": "1", "role": "user", "sender_name": "Alice", "content": "Find cafes in Indiranagar.", "visibility": "shared"},
                {"id": "2", "role": "assistant", "content": "Cafe A and Cafe B are good options.", "visibility": "shared"},
                {"id": "3", "role": "user", "sender_name": "Bob", "content": "Which one has Wi-Fi?", "visibility": "shared"},
            ],
            user_id="bob",
        )
        block = prompt_block(context)
        self.assertIn("Alice: Find cafes", block)
        self.assertIn("Bob: Which one has Wi-Fi?", block)
        self.assertIn("Nexa: Cafe A", block)

    def test_private_visible_message_does_not_enter_shared_context_for_other_user(self) -> None:
        context = derive_context(
            [
                {"id": "1", "role": "user", "sender_name": "Alice", "content": "Shared plan", "visibility": "shared"},
                {"id": "2", "role": "assistant", "content": "Alice private result", "visibility": "private", "visible_to_user_id": "alice"},
                {"id": "3", "role": "assistant", "content": "Shared result", "visibility": "shared"},
            ],
            user_id="bob",
        )
        block = prompt_block(context)
        self.assertNotIn("Alice private result", block)
        self.assertIn("Shared result", block)

    def test_follow_up_routing_continues_previous_research_workflow(self) -> None:
        session = {"last_workflow": "research", "last_domains": ["web"]}
        decision = route_request("Is it open tomorrow?", session_context=session)
        self.assertEqual(decision.workflow, Workflow.RESEARCH)
        self.assertEqual(decision.domains, ["web"])

    def test_follow_up_signal_does_not_copy_old_transcript_into_routing(self) -> None:
        session = {"last_workflow": "research", "last_domains": ["web"]}
        routed = follow_up_query("Compare it with the second one", session)
        self.assertIn("Compare it with the second one", routed)
        self.assertIn("workflow=research", routed)
        self.assertNotIn("password", routed)

    def test_agent_context_uses_session_context_when_available(self) -> None:
        with patch(
            "Backend.AgentArchitecture.load_for_agent",
            return_value={
                "recent": "Alice: We chose Cafe B.",
                "summary": "Earlier visible session excerpts",
                "entities": [],
                "active_task": "Check Cafe B tomorrow",
            },
        ):
            context = build_context()
        self.assertIn("Cafe B", context.conversation)
        self.assertEqual(context.session_context["active_task"], "Check Cafe B tomorrow")


if __name__ == "__main__":
    unittest.main()

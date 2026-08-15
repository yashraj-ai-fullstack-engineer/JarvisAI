from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from Backend.JarvisAgent import _bounded_prompt_messages, _parse_json_object, _plan_validation_error


class PromptWindowTests(unittest.TestCase):
    def test_preserves_user_message_after_many_tool_messages(self) -> None:
        messages = [HumanMessage(content="Read my latest three emails")]
        for index in range(12):
            messages.append(AIMessage(content="", tool_calls=[{
                "name": "gmail_search_messages",
                "args": {},
                "id": f"call-{index}",
                "type": "tool_call",
            }]))
            messages.append(ToolMessage(content="{}", tool_call_id=f"call-{index}"))

        bounded = _bounded_prompt_messages(messages)

        self.assertTrue(any(isinstance(message, HumanMessage) for message in bounded))
        self.assertEqual(bounded[0].content, "Read my latest three emails")

    def test_perception_json_parser_rejects_non_json(self) -> None:
        self.assertIsNone(_parse_json_object("not a plan"))
        self.assertEqual(_parse_json_object('{"intent":"gmail"}'), {"intent": "gmail"})

    def test_plan_validation_rejects_unknown_tools(self) -> None:
        plan = {
            "intent": "read email",
            "needs_tools": True,
            "tool_names": ["not_a_tool"],
            "workflow": ["read"],
            "max_tool_calls": 2,
        }
        self.assertIn("not available", _plan_validation_error(plan, ["gmail_search_messages"]))

    def test_plan_validation_does_not_accept_tool_aliases(self) -> None:
        plan = {
            "intent": "research",
            "needs_tools": True,
            "tool_names": ["web_search"],
            "workflow": ["research"],
            "max_tool_calls": 1,
        }
        self.assertIn("not available", _plan_validation_error(plan, ["research_web"]))


if __name__ == "__main__":
    unittest.main()

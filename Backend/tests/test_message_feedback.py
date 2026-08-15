from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import Backend.MongoStore as store


class MessageFeedbackTests(TestCase):
    def test_feedback_updates_the_viewer_state_and_appends_an_event(self) -> None:
        message = {
            "id": "assistant-message",
            "session_id": "session-a",
            "role": "assistant",
            "content": "An answer",
            "created_at": datetime.now(timezone.utc),
            "feedback_by_user": {"user-a": {"reaction": "like"}},
        }
        database = SimpleNamespace(
            chat_messages=MagicMock(find_one_and_update=MagicMock(return_value=message)),
            message_feedback=MagicMock(),
        )

        with patch("Backend.MongoStore._db", return_value=database), patch(
            "Backend.MongoStore._visible_message_query", return_value={"session_id": "session-a"}
        ):
            result = store.set_message_feedback("session-a", "assistant-message", "user-a", "like")

        self.assertEqual(result["feedback"], "like")
        self.assertEqual(database.chat_messages.find_one_and_update.call_count, 1)
        event = database.message_feedback.insert_one.call_args.args[0]
        self.assertEqual(event["reaction"], "like")
        self.assertEqual(event["message_id"], "assistant-message")
        self.assertEqual(event["user_id"], "user-a")

    def test_feedback_rejects_non_assistant_reactions(self) -> None:
        with self.assertRaisesRegex(ValueError, "either 'like' or 'dislike'"):
            store.set_message_feedback("session-a", "message", "user-a", "skip")

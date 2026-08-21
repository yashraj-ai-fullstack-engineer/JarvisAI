from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from Backend.WebApp import app


USER = {"id": "user-a", "name": "Asha", "email": "asha@example.com"}
SESSION_ID = "12345678-1234-1234-1234-123456789012"


class PersonaAPITests(unittest.TestCase):
    def test_create_run_is_self_only_idempotent_command_endpoint(self) -> None:
        run = {"id": "run-a", "status": "queued", "progress": 0}
        with TestClient(app) as browser, patch("Backend.WebApp._authenticated_user", return_value=USER), patch("Backend.WebApp.queue_persona_run", return_value=run) as queue:
            response = browser.post("/api/persona/runs", json={"user_id": "user-b"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["persona_url"], "/persona")
        self.assertIn("no-store", response.headers["cache-control"])
        queue.assert_called_once_with("user-a")

    def test_persona_dashboard_is_private_and_does_not_cache(self) -> None:
        snapshot = {"profile": {"persona_name": "Thoughtful Builder"}, "run": None, "is_processing": False}
        with TestClient(app) as browser, patch("Backend.WebApp._authenticated_user", return_value=USER), patch("Backend.WebApp.persona_snapshot", return_value=snapshot) as load:
            response = browser.get("/api/persona")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), snapshot)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        load.assert_called_once_with("user-a")

    def test_unauthenticated_persona_request_is_rejected(self) -> None:
        with TestClient(app) as browser, patch("Backend.WebApp._authenticated_user", side_effect=HTTPException(status_code=401, detail="Sign in to continue.")):
            response = browser.get("/api/persona")
        self.assertEqual(response.status_code, 401)

    def test_me_bypasses_shared_message_persistence_and_broadcast(self) -> None:
        run = {"id": "run-a", "status": "queued"}
        with TestClient(app) as browser, patch("Backend.WebApp._require_chat_session", return_value=USER), patch("Backend.WebApp.queue_persona_run", return_value=run) as queue, patch("Backend.WebApp.save_message") as save:
            response = browser.post(f"/api/chats/{SESSION_ID}/messages", json={"message": "/me"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["persona_url"], "/persona")
        queue.assert_called_once_with("user-a")
        save.assert_not_called()

    def test_me_bypasses_agent_stream_before_session_mode_checks(self) -> None:
        run = {"id": "run-a", "status": "queued"}
        with TestClient(app) as browser, patch("Backend.WebApp._require_chat_session", return_value=USER), patch("Backend.WebApp.queue_persona_run", return_value=run), patch("Backend.WebApp.active_participant_count") as participants, patch("Backend.WebApp.save_message") as save:
            response = browser.post("/api/chat/stream", json={"message": "@Nexa /me", "session_id": SESSION_ID})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["run"]["id"], "run-a")
        participants.assert_not_called()
        save.assert_not_called()

    def test_me_sent_by_legacy_shared_room_socket_is_private(self) -> None:
        run = {"id": "run-a", "status": "queued"}
        with TestClient(app) as browser, patch("Backend.WebApp.auth_user", return_value=USER), patch("Backend.WebApp.owns_chat_session", return_value=True), patch("Backend.WebApp.queue_persona_run", return_value=run), patch("Backend.WebApp.save_message") as save:
            with browser.websocket_connect(f"/api/chats/{SESSION_ID}/live?auth_token=test") as socket:
                socket.send_json({"type": "message", "content": "/me"})
                event = socket.receive_json()
        self.assertEqual(event["type"], "persona_queued")
        self.assertEqual(event["run"]["id"], "run-a")
        save.assert_not_called()

    def test_delete_persona_can_only_target_authenticated_user(self) -> None:
        with TestClient(app) as browser, patch("Backend.WebApp._authenticated_user", return_value=USER), patch("Backend.WebApp.delete_persona") as delete:
            response = browser.request("DELETE", "/api/persona", json={"user_id": "user-b"})
        self.assertEqual(response.status_code, 200)
        delete.assert_called_once_with("user-a")

    def test_command_grammar_is_exact(self) -> None:
        from Backend.WebApp import _is_persona_command

        self.assertTrue(_is_persona_command('/me'))
        self.assertTrue(_is_persona_command('@Nexa /me'))
        self.assertFalse(_is_persona_command('/membership'))
        self.assertFalse(_is_persona_command('/me please analyze me'))


if __name__ == "__main__":
    unittest.main()

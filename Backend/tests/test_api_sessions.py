from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import Backend.Chatbot as chatbot
import Backend.EmailManager as email_manager
import Backend.MCPManager as mcp_manager
from Backend.WebApp import app


class BrowserSessionAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.patches = [
            patch.object(chatbot, "SESSION_DATA_DIR", root / "sessions"),
            patch.object(
                email_manager,
                "EMAIL_RECORDS_PATH",
                root / "PendingEmails.json",
            ),
            patch.object(
                mcp_manager,
                "PENDING_ACTIONS_PATH",
                root / "PendingMCPActions.json",
            ),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_public_capabilities_do_not_expose_server_credentials(self) -> None:
        with TestClient(app) as browser:
            response = browser.get("/api/mcp/servers")
        self.assertEqual(response.status_code, 200)
        serialized = response.text.lower()
        self.assertNotIn("authorization", serialized)
        self.assertNotIn("client_secret", serialized)
        self.assertNotIn('"headers"', serialized)
        self.assertNotIn('"env"', serialized)

    def test_feedback_put_preflight_is_allowed(self) -> None:
        with TestClient(app) as browser:
            response = browser.options(
                "/api/chats/8e584ee2-ed42-41d3-896a-66493cf72c12/messages/695a7b05-bd33-4b43-b85f-aae26913a608/feedback",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "PUT",
                    "Access-Control-Request-Headers": "authorization,content-type",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("PUT", response.headers.get("access-control-allow-methods", ""))


if __name__ == "__main__":
    unittest.main()

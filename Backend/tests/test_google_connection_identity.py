from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import Backend.GoogleOAuth as google_oauth


class GoogleConnectionIdentityTests(unittest.TestCase):
    def test_legacy_browser_record_migrates_only_for_the_matching_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            google_oauth, "DATA_PATH", Path(temp_dir) / "connections.json"
        ):
            google_oauth._write_json(google_oauth.DATA_PATH, [{
                "session_id": "legacy-browser",
                "service": "gmail",
                "email": "user@example.com",
                "token": "encrypted-token-not-read-in-this-test",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }])

            record = google_oauth._get_record("legacy-browser", "gmail", "user-1")
            self.assertEqual(record["user_id"], "user-1")
            persisted = json.loads(google_oauth.DATA_PATH.read_text(encoding="utf-8"))
            self.assertEqual(persisted[0]["user_id"], "user-1")
            self.assertIsNone(google_oauth._get_record("other-browser", "gmail", "user-2"))

    def test_user_owned_record_is_available_without_the_original_browser_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            google_oauth, "DATA_PATH", Path(temp_dir) / "connections.json"
        ):
            google_oauth._write_json(google_oauth.DATA_PATH, [{
                "user_id": "user-1",
                "session_id": "old-browser",
                "service": "gmail",
                "email": "user@example.com",
                "token": "encrypted-token-not-read-in-this-test",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }])
            with google_oauth.google_user_context("user-1"):
                self.assertTrue(google_oauth.google_mcp_connected("gmail"))
                status = google_oauth.service_status("new-browser", "user-1")
            self.assertTrue(next(item for item in status if item["service"] == "gmail")["connected"])


if __name__ == "__main__":
    unittest.main()

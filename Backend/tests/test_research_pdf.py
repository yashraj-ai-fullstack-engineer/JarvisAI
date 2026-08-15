from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from pypdf import PdfReader

from Backend.DeepResearch import ResearchRunStore, build_research_plan
from Backend.MongoStore import chat_session_context
from Backend.ResearchPdfService import _LocalStorage, create_or_reuse_export, local_export_path


class ResearchPdfTests(TestCase):
    def _completed_run(self) -> dict:
        return {
            "id": "run-pdf-123",
            "user_id": "user-a",
            "session_id": "session-a",
            "status": "completed",
            "topic": "software and AI",
            "question": "Compare two agent frameworks.",
            "completed_at": "2026-08-15T12:00:00+00:00",
            "source_ids": ["github", "public_web"],
            "source_errors": ["One secondary source was rate-limited."],
            "report": "## Answer\nThe first framework has a larger public repository. [S1]\n\n## Sources\n- [S1] [Example repository](https://github.com/example/project)",
            "evidence_manifest": [{
                "reference": "S1",
                "source_id": "github",
                "source_label": "GitHub",
                "title": "Example repository",
                "url": "https://github.com/example/project",
                "published_at": "2026-08-14",
            }],
        }

    def test_local_export_is_cached_and_contains_report_and_source_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch("Backend.ResearchPdfService.DATA_DIR", Path(temp_dir)):
            run = self._completed_run()
            first = create_or_reuse_export(run)
            self.assertEqual(first["storage"], "local")
            self.assertTrue(local_export_path(first).is_file())
            self.assertNotIn("user-a", first["storage_key"])

            reader = PdfReader(str(local_export_path(first)))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            self.assertIn("Nexa Research Report", extracted)
            self.assertIn("Verified source trace", extracted)
            self.assertIn("Example repository", extracted)

            run["pdf_export"] = first
            self.assertEqual(create_or_reuse_export(run), first)

    def test_local_storage_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch("Backend.ResearchPdfService.DATA_DIR", Path(temp_dir)):
            storage = _LocalStorage()
            with self.assertRaisesRegex(Exception, "Invalid private PDF storage"):
                storage.local_path("../another-user/report.pdf")

    def test_research_run_pdf_metadata_is_user_and_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchRunStore(Path(temp_dir) / "runs.json", persist_to_mongo=False)
            plan = build_research_plan("Explain the history of the Internet", {"internet_archive", "dbpedia", "public_web"})
            with chat_session_context("session-a", "user-a"):
                run = store.start("Explain the history of the Internet", plan)
                store.finish(run.id, report="## Answer\nA cited report")
                store.set_pdf_export(run.id, {"storage": "local", "storage_key": "research-exports/x/report.pdf"})
                owned = store.get_for_current_user(run.id)
            with chat_session_context("session-a", "user-b"):
                other_user = store.get_for_current_user(run.id)
            with chat_session_context("session-b", "user-a"):
                other_session = store.get_for_current_user(run.id)

        self.assertEqual(owned["pdf_export"]["storage"], "local")
        self.assertIsNone(other_user)
        self.assertIsNone(other_session)

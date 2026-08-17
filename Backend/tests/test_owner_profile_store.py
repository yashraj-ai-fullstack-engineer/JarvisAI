from __future__ import annotations

import asyncio
from unittest import TestCase
from unittest.mock import MagicMock, patch

from Backend.AgentArchitecture import Workflow, route_request
from Backend.JarvisAgent import _perceive_request_impl
import Backend.OwnerRAG as owner_rag


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class OwnerProfileStoreTests(TestCase):
    def test_owner_markers_route_to_private_knowledge_workflow(self) -> None:
        for query in (
            "Who is Yashraj?",
            "Tell me about your owner",
            "Who is your master?",
        ):
            self.assertTrue(owner_rag.is_owner_question(query), query)
            self.assertEqual(route_request(query).workflow, Workflow.KNOWLEDGE, query)

    def test_owner_plan_is_deterministic_and_uses_only_resume_tool(self) -> None:
        plan = asyncio.run(
            _perceive_request_impl(
                "Who is your master?",
                [FakeTool("get_capabilities"), FakeTool("answer_owner_profile")],
            )
        )
        self.assertTrue(plan["needs_tools"])
        self.assertEqual(plan["tool_names"], ["answer_owner_profile"])
        self.assertEqual(plan["max_tool_calls"], 1)

    def test_supabase_retrieval_uses_vector_rpc(self) -> None:
        response = MagicMock()
        response.ok = True
        response.json.return_value = [
            {
                "chunk_key": "resume-p1-profile-owner-identity-and-summary",
                "chunk_index": 0,
                "page_number": 1,
                "section": "Profile",
                "title": "Owner identity and summary",
                "chunk_text": "Yashraj Gupta is an AI Engineer.",
                "similarity": 0.93,
            }
        ]
        with (
            patch.object(owner_rag, "sync_owner_profile", return_value={"ok": True}),
            patch.object(owner_rag, "embed_pdf_texts", return_value=[[0.1] * 2048]),
            patch.object(owner_rag, "supabase_url", return_value="https://example.supabase.co"),
            patch.object(owner_rag, "supabase_headers", return_value={"Authorization": "Bearer test"}),
            patch("requests.post", return_value=response) as post,
        ):
            result = owner_rag._retrieve_supabase_context("Who is Yashraj?", 4)

        self.assertEqual(result["retrieval_mode"], "supabase_vector")
        self.assertEqual(result["matches"][0]["page"], 1)
        self.assertIn("match_owner_profile_chunks", post.call_args.args[0])

    def test_local_lexical_fallback_never_needs_embedding_server(self) -> None:
        index = {
            "source_name": "Resume_Yashraj.pdf",
            "embedding_model": "lexical-fallback",
            "chunks": [
                {
                    "id": "profile",
                    "page": 1,
                    "section": "Profile",
                    "title": "Owner identity and summary",
                    "text": "Yashraj Gupta is an AI Engineer building LLM applications.",
                    "order": 0,
                },
                {
                    "id": "education",
                    "page": 1,
                    "section": "Education",
                    "title": "Education",
                    "text": "Bachelor of Technology in Computer Science.",
                    "order": 1,
                },
            ],
        }
        with (
            patch.object(owner_rag, "_supabase_enabled", return_value=False),
            patch.object(owner_rag, "load_owner_index", return_value=index),
        ):
            result = owner_rag.retrieve_owner_context("Who is Yashraj?")

        self.assertEqual(result["retrieval_mode"], "local_lexical_fallback")
        self.assertEqual(result["matches"][0]["id"], "profile")


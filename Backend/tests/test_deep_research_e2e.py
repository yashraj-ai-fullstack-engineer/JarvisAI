from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import Backend.DeepResearch as research
from Backend.DeepResearch import ResearchEvidence, ResearchRunStore, SourceResult
from Backend.MongoStore import chat_session_context


class DeepResearchEndToEndTests(TestCase):
    """Deterministic full-pipeline tests for the /research SSE workflow."""

    cases = (
        (
            "Compare Maggi ingredients and allergens",
            ("open_food_facts", "public_web"),
        ),
        (
            "Map A R Rahman's albums and collaborations",
            ("musicbrainz", "dbpedia", "public_web"),
        ),
        (
            "Compare LLM agent frameworks and their release activity",
            ("github", "github_releases", "huggingface", "stackexchange", "hacker_news", "public_web"),
        ),
        (
            "How has https://example.com changed over time?",
            ("internet_archive", "common_crawl", "dbpedia", "public_web"),
        ),
        (
            "What is the current climate change news coverage?",
            ("gdelt", "rss", "public_web"),
        ),
        (
            "Analyse high performing stocks in India and the US market",
            ("financial_market_data", "yahoo_finance"),
        ),
        (
            "whatt are the trending responsitories in the recent times",
            ("github",),
        ),
    )

    def test_each_topic_completes_with_only_the_validated_source_plan(self) -> None:
        def fake_fetch(source_id: str, _: str) -> SourceResult:
            return SourceResult(
                source_id=source_id,
                evidence=[ResearchEvidence(
                    reference="",
                    source_id=source_id,
                    source_label=source_id,
                    title=f"{source_id} evidence",
                    url=f"https://e2e.example/{source_id}",
                    summary="Controlled end-to-end evidence.",
                )],
            )

        async def collect(question: str) -> list[dict]:
            return [
                event
                async for event in research.DeepResearchStream(question, history_query=f"/research {question}")
            ]

        live_sources = {source_id: research.SOURCES_BY_ID[source_id] for source_id in research.SOURCES_BY_ID}
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchRunStore(Path(temp_dir) / "runs.json", persist_to_mongo=False)
            with chat_session_context("research-e2e-session", "research-e2e-user"), patch("Backend.DeepResearch.RESEARCH_RUN_STORE", store), patch("Backend.DeepResearch._live_sources", return_value=live_sources), patch("Backend.DeepResearch.fetch_source", side_effect=fake_fetch), patch("Backend.DeepResearch._compose_report", return_value="## Answer\nSupported finding [S1]"), patch("Backend.DeepResearch.SaveExchange"):
                for question, expected_sources in self.cases:
                    events = asyncio.run(collect(question))
                    plan = next(event for event in events if event["type"] == "research_plan")
                    self.assertEqual(tuple(source["id"] for source in plan["sources"]), expected_sources)
                    completed = [event for event in events if event.get("message", "").endswith("evidence ready")]
                    self.assertEqual(len(completed), len(expected_sources))
                    self.assertEqual(events[-1]["type"], "done")
                    self.assertIn("[S1]", events[-1]["answer"])

"""Opt-in live contract checks for free public research providers.

Run manually or in scheduled CI with NEXA_RUN_LIVE_RESEARCH_TESTS=true. They
are intentionally skipped during normal tests because free endpoints can be
temporarily rate-limited or unavailable.
"""

from __future__ import annotations

import os
import unittest

from Backend.DeepResearch import fetch_source


@unittest.skipUnless(
    os.getenv("NEXA_RUN_LIVE_RESEARCH_TESTS", "").lower() == "true",
    "Set NEXA_RUN_LIVE_RESEARCH_TESTS=true to call public research providers.",
)
class LiveResearchProviderContractTests(unittest.TestCase):
    cases = (
        ("github", "LangChain"),
        ("github_releases", "LangChain"),
        ("stackexchange", "FastAPI"),
        ("hacker_news", "Show Hacker News top stories"),
        ("huggingface", "llama"),
        ("gdelt", "climate change"),
        ("internet_archive", "internet"),
        ("open_food_facts", "Maggi"),
        ("musicbrainz", "A R Rahman"),
        ("dbpedia", "Albert Einstein"),
        ("common_crawl", "How has https://example.com changed over time?"),
        ("public_web", "OpenAI official announcements"),
    )

    def test_provider_adapters_return_normalized_evidence(self) -> None:
        for source_id, question in self.cases:
            with self.subTest(source=source_id):
                result = fetch_source(source_id, question)
                if result.error_kind == "rate_limited":
                    # This verifies Nexa classified a real provider throttle
                    # correctly; it is not a malformed adapter response.
                    self.assertEqual(result.status_code, 429)
                    continue
                self.assertFalse(result.error, result.error)
                self.assertGreater(len(result.evidence), 0, "No evidence returned")
                for evidence in result.evidence:
                    self.assertEqual(evidence.source_id, source_id)
                    self.assertTrue(evidence.title)
                    self.assertTrue(evidence.summary)
                    self.assertTrue(evidence.url.startswith(("https://", "http://")))

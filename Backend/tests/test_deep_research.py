from __future__ import annotations

import asyncio
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import requests

import Backend.DeepResearch as research
from Backend.DeepResearch import (
    DeepResearchStream,
    ResearchEvidence,
    ResearchError,
    ResearchRunStore,
    SourceResult,
    build_research_plan,
    parse_research_command,
)
from Backend.AgentTools import _emit_status
from Backend.MongoStore import chat_session_context


class DeepResearchTests(TestCase):
    def test_research_progress_emitter_is_safe_outside_langgraph(self) -> None:
        # /research reuses the bounded public-web implementation directly,
        # rather than via a LangGraph ToolNode.
        _emit_status("Testing", "Research", "No graph runtime is active.")

    def test_research_command_requires_a_question(self) -> None:
        self.assertEqual(parse_research_command("/research"), "")
        self.assertEqual(parse_research_command(" /research   climate migration "), "climate migration")
        self.assertIsNone(parse_research_command("research climate migration"))

    def test_food_plan_selects_food_evidence_not_technical_sources(self) -> None:
        plan = build_research_plan(
            "Compare the ingredients and allergens in Maggi and Yippee noodles",
            {"open_food_facts", "public_web", "github", "stackexchange"},
        )
        self.assertEqual(plan.topic, "food and nutrition")
        self.assertEqual(plan.source_ids, ["open_food_facts", "public_web"])
        self.assertNotIn("github", plan.source_ids)

    def test_software_plan_only_selects_sources_in_live_closed_set(self) -> None:
        plan = build_research_plan(
            "Compare LangGraph and CrewAI release activity and developer issues",
            {"github", "github_releases", "public_web"},
        )
        self.assertEqual(plan.source_ids, ["github", "github_releases", "public_web"])
        self.assertNotIn("stackexchange", plan.source_ids)
        self.assertNotIn("hacker_news", plan.source_ids)

    def test_general_research_sets_a_depth_target_for_the_single_web_source(self) -> None:
        plan = build_research_plan(
            "Explain the causes, effects, and competing explanations for urban heat islands",
            {"public_web"},
        )

        self.assertEqual(plan.topic, "general research")
        self.assertEqual(plan.evidence_target, 8)
        self.assertEqual(plan.source_coverage_target, 1)

    def test_coverage_pass_only_reuses_sources_approved_by_the_plan(self) -> None:
        plan = build_research_plan(
            "Explain the causes and effects of urban heat islands",
            {"public_web", "github", "gdelt"},
        )
        initial_evidence = [ResearchEvidence(
            reference="S1",
            source_id="public_web",
            source_label="Public web",
            title="Evidence",
            url="https://example.com/evidence",
            summary="Initial result.",
        )]

        coverage_sources = research._coverage_pass_sources(plan, initial_evidence)

        self.assertEqual(coverage_sources, ["public_web"])
        self.assertNotIn("github", coverage_sources)
        self.assertNotIn("gdelt", coverage_sources)

    def test_common_crawl_requires_an_explicit_public_https_url(self) -> None:
        with_url = build_research_plan(
            "How has https://example.com changed over time?",
            {"internet_archive", "common_crawl", "dbpedia", "public_web"},
        )
        without_url = build_research_plan(
            "How has this company changed over time?",
            {"internet_archive", "common_crawl", "dbpedia", "public_web"},
        )
        self.assertIn("common_crawl", with_url.source_ids)
        self.assertNotIn("common_crawl", without_url.source_ids)

    def test_research_store_is_user_scoped_and_persists_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchRunStore(Path(temp_dir) / "research.json", persist_to_mongo=False)
            plan = build_research_plan("Explain the history of the Internet", {"internet_archive", "dbpedia", "public_web"})
            with chat_session_context("session-a", "user-a"):
                run = store.start("Explain the history of the Internet", plan)
                store.finish(run.id, evidence_count=3, report="## Answer\nA cited report")
                visible = store.list_for_current_user()
            with chat_session_context("session-a", "user-b"):
                hidden = store.list_for_current_user()
            payload = json.loads((Path(temp_dir) / "research.json").read_text(encoding="utf-8"))

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["report"], "## Answer\nA cited report")
        self.assertEqual(hidden, [])
        self.assertEqual(payload[0]["status"], "completed")

    def test_stream_uses_the_validated_source_plan_and_returns_a_cited_report(self) -> None:
        def fake_fetch(source_id: str, question: str) -> SourceResult:
            evidence = ResearchEvidence(
                reference="",
                source_id=source_id,
                source_label=source_id,
                title=f"Evidence from {source_id}",
                url=f"https://example.com/{source_id}",
                summary="Bounded test evidence.",
            )
            return SourceResult(source_id=source_id, evidence=[evidence])

        async def collect_events() -> list[dict]:
            events = []
            async for event in DeepResearchStream(
                "Compare ingredients in Maggi noodles",
                history_query="/research Compare ingredients in Maggi noodles",
            ):
                events.append(event)
            return events

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchRunStore(Path(temp_dir) / "research.json", persist_to_mongo=False)
            with chat_session_context("session-a", "user-a"), patch("Backend.DeepResearch.RESEARCH_RUN_STORE", store), patch("Backend.DeepResearch.fetch_source", side_effect=fake_fetch), patch("Backend.DeepResearch._compose_report", return_value="## Answer\nCited finding [S1]"), patch("Backend.DeepResearch.SaveExchange"):
                events = asyncio.run(collect_events())

        plan_event = next(event for event in events if event["type"] == "research_plan")
        self.assertEqual([source["id"] for source in plan_event["sources"]], ["open_food_facts", "public_web"])
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("[S1]", events[-1]["answer"])

    def test_transient_provider_failure_is_not_cached(self) -> None:
        def unavailable(_: str):
            raise RuntimeError("temporary provider outage")

        def recovered(_: str) -> list[ResearchEvidence]:
            return [ResearchEvidence(
                reference="",
                source_id="public_web",
                source_label="Public web",
                title="Recovered evidence",
                url="https://example.com/recovered",
                summary="The provider recovered.",
            )]

        with patch("Backend.DeepResearch.RESEARCH_CACHE", research.ResearchCache()), patch.dict("Backend.DeepResearch.ADAPTERS", {"public_web": unavailable}):
            first = research.fetch_source("public_web", "unique recovery test")
            self.assertTrue(first.error)
            with patch.dict("Backend.DeepResearch.ADAPTERS", {"public_web": recovered}):
                second = research.fetch_source("public_web", "unique recovery test")

        self.assertFalse(second.error)
        self.assertEqual(second.evidence[0].title, "Recovered evidence")

    def test_provider_rate_limit_is_classified_without_exposing_request_details(self) -> None:
        def rate_limited(_: str):
            error = requests.HTTPError("provider rejected request")
            error.response = MagicMock(status_code=429)
            raise error

        with patch("Backend.DeepResearch.RESEARCH_CACHE", research.ResearchCache()), patch.dict("Backend.DeepResearch.ADAPTERS", {"public_web": rate_limited}):
            result = research.fetch_source("public_web", "unique rate limit test")

        self.assertEqual(result.error_kind, "rate_limited")
        self.assertEqual(result.status_code, 429)
        self.assertEqual(result.error, "Public web is temporarily rate-limited.")

    def test_open_food_facts_uses_its_official_mirror_after_primary_outage(self) -> None:
        product = {
            "code": "123",
            "product_name": "Test noodles",
            "ingredients_text": "wheat, salt",
            "nutriments": {"energy-kcal_100g": 120},
        }
        with patch(
            "Backend.DeepResearch._request_json",
            side_effect=[requests.HTTPError("temporary 503"), {"products": [product]}],
        ) as request_json:
            evidence = research._open_food_facts_search("Test noodles")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].title, "Test noodles")
        self.assertEqual(request_json.call_count, 2)

    def test_stock_research_uses_only_a_registered_market_data_source(self) -> None:
        question = "Can you analyse high performing stocks in India and the US market?"
        plan = build_research_plan(question, {"financial_market_data", "public_web", "gdelt"})

        self.assertEqual(plan.topic, "financial markets")
        self.assertEqual(plan.source_ids, ["financial_market_data"])
        self.assertNotIn("public_web", plan.source_ids)

    def test_stock_research_can_select_yahoo_wrapper_without_an_api_key(self) -> None:
        plan = build_research_plan(
            "Analyse high performing stocks in India and the US market",
            {"yahoo_finance", "public_web"},
        )

        self.assertEqual(plan.source_ids, ["yahoo_finance"])

    def test_stock_research_never_downgrades_to_general_web_when_market_data_is_missing(self) -> None:
        question = "Can you analyse high performing stocks in India and the US market?"

        with self.assertRaisesRegex(ResearchError, "will not substitute ordinary web-search results"):
            build_research_plan(question, {"public_web", "gdelt"})

    def test_public_web_finance_gate_rejects_irrelevant_search_collisions(self) -> None:
        raw = json.dumps({
            "ok": True,
            "sources": [
                {"title": "Unsupported client – Canva", "url": "https://www.canva.in/", "excerpt": "Browser support."},
                {"title": "CAN Definition & Meaning", "url": "https://www.merriam-webster.com/dictionary/can", "excerpt": "Dictionary definition."},
                {"title": "NSE market data policy", "url": "https://www.nseindia.com/static/market-data/nse-data-policy", "excerpt": "Market data policy for the stock exchange."},
            ],
        })
        fake_tool = MagicMock()
        fake_tool.invoke.return_value = raw

        with patch("Backend.DeepResearch.research_web", fake_tool):
            evidence = research._public_web_search("Analyse high performing stocks in India and the US market")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].title, "NSE market data policy")

    def test_market_data_adapter_keeps_india_and_us_records_and_hides_api_key(self) -> None:
        response = lambda symbol, name, change, price: [{
            "symbol": symbol,
            "name": name,
            "changesPercentage": change,
            "price": price,
            "currency": "USD" if symbol != "RELIANCE.BO" else "INR",
            "timestamp": "2026-08-15T00:00:00Z",
        }]
        with patch("Backend.DeepResearch.get_config", return_value="test-secret-key"), patch(
            "Backend.DeepResearch._request_json",
            side_effect=[
                response("AAPL", "Apple", 3.5, 230),
                response("MSFT", "Microsoft", 2.0, 510),
                response("RELIANCE.BO", "Reliance Industries", 1.5, 1550),
            ],
        ):
            evidence = research._financial_market_data_search(
                "Analyse high performing stocks in India and the US market"
            )

        self.assertEqual(len(evidence), 3)
        self.assertEqual({item.metadata["market"] for item in evidence}, {"United States / NASDAQ", "United States / NYSE", "India / BSE"})
        self.assertTrue(all("test-secret-key" not in item.url for item in evidence))
        self.assertEqual(len(research._deduplicate_evidence(evidence)), 3)

    def test_yahoo_finance_wrapper_needs_no_api_key_and_keeps_market_scope(self) -> None:
        equity_query = MagicMock(side_effect=lambda operator, operand: {"operator": operator, "operand": operand})
        screen = MagicMock(side_effect=[
            {"quotes": [{
                "symbol": "AAPL", "shortName": "Apple", "regularMarketChangePercent": 2.5,
                "regularMarketPrice": 230, "currency": "USD", "regularMarketTime": "2026-08-15T00:00:00Z",
            }]},
            {"quotes": [{
                "symbol": "RELIANCE.NS", "shortName": "Reliance Industries", "regularMarketChangePercent": 1.5,
                "regularMarketPrice": 1550, "currency": "INR", "regularMarketTime": "2026-08-15T00:00:00Z",
            }]},
        ])
        fake_yahoo = SimpleNamespace(EquityQuery=equity_query, screen=screen)

        with patch("Backend.DeepResearch.importlib.import_module", return_value=fake_yahoo):
            evidence = research._yahoo_finance_search(
                "Analyse high performing stocks in India and the US market"
            )

        self.assertEqual(len(evidence), 2)
        self.assertEqual({item.metadata["market"] for item in evidence}, {"United States / Yahoo Finance screen", "India / Yahoo Finance screen"})
        self.assertTrue(all(item.metadata["source_status"] == "unofficial community wrapper" for item in evidence))
        self.assertTrue(all(item.url.startswith("https://finance.yahoo.com/quote/") for item in evidence))

    def test_trending_repository_typo_routes_only_to_github(self) -> None:
        plan = build_research_plan(
            "whatt are the trending responsitories in the recent times",
            {"github", "github_releases", "hacker_news", "public_web"},
        )

        self.assertEqual(plan.topic, "GitHub repository activity")
        self.assertEqual(plan.source_ids, ["github"])
        self.assertIn("last 30 days", " ".join(plan.constraints))

    def test_github_trending_request_does_not_send_user_prose_to_api_search(self) -> None:
        data = {"items": [{
            "full_name": "example/trending-project", "html_url": "https://github.com/example/trending-project",
            "description": "A new popular project.", "updated_at": "2026-08-15T00:00:00Z",
            "stargazers_count": 100, "forks_count": 10, "language": "Python", "open_issues_count": 2,
        }]}
        with patch("Backend.DeepResearch._request_json", return_value=data) as request_json:
            evidence = research._github_search("whatt are the trending responsitories in the recent times")

        params = request_json.call_args.kwargs["params"]
        self.assertRegex(params["q"], r"^created:>=\d{4}-\d{2}-\d{2}$")
        self.assertEqual(params["sort"], "stars")
        self.assertEqual(evidence[0].metadata["mode"], "new_repositories_ranked_by_stars")

    def test_github_403_rate_limit_is_not_misreported_as_a_credential_error(self) -> None:
        def rate_limited(_: str):
            error = requests.HTTPError("GitHub search quota exhausted")
            error.response = MagicMock(status_code=403, headers={"X-RateLimit-Remaining": "0"})
            raise error

        with patch("Backend.DeepResearch.RESEARCH_CACHE", research.ResearchCache()), patch.dict("Backend.DeepResearch.ADAPTERS", {"github": rate_limited}):
            result = research.fetch_source("github", "unique github rate limit test")

        self.assertEqual(result.error_kind, "rate_limited")
        self.assertEqual(result.error, "GitHub is temporarily rate-limited.")

    def test_composer_requests_a_research_only_output_budget_and_deep_structure(self) -> None:
        plan = build_research_plan(
            "Explain the causes and effects of urban heat islands",
            {"public_web"},
        )
        evidence = [ResearchEvidence(
            reference="S1",
            source_id="public_web",
            source_label="Public web",
            title="Evidence",
            url="https://example.com/evidence",
            summary="A bounded evidence record.",
        )]
        response = "## Answer\nSupported finding [S1]\n## Sources\n"

        with patch("Backend.DeepResearch.generate_text", return_value=response) as generate:
            report = research._compose_report("Explain urban heat islands", plan, evidence, [])

        prompt = generate.call_args.args[0]
        self.assertIn("## Analysis and interpretation", prompt)
        self.assertIn("## Counterevidence and open questions", prompt)
        self.assertEqual(
            generate.call_args.args[5], research.RESEARCH_SYNTHESIS_MAX_OUTPUT_TOKENS
        )
        self.assertIn("## Sources", report)

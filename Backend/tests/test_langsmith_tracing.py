from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import Backend.LangSmithTracing as tracing


class LangSmithTracingTests(TestCase):
    def test_request_descriptor_preserves_the_query_for_root_trace_metadata(self) -> None:
        raw_request = "summarize my private email from alice@example.com"

        descriptor = tracing.request_descriptor(raw_request)

        self.assertEqual(descriptor, {"query": raw_request})

    def test_tracing_requires_an_explicit_toggle_and_api_key(self) -> None:
        with patch("Backend.LangSmithTracing.get_config", side_effect=lambda name, default="": {
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_API_KEY": "configured-key",
        }.get(name, default)):
            self.assertFalse(tracing.tracing_enabled())

        with patch("Backend.LangSmithTracing.get_config", side_effect=lambda name, default="": {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "",
        }.get(name, default)):
            self.assertFalse(tracing.tracing_enabled())

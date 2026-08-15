from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

import Backend.LLMProvider as provider


class LLMProviderOutputBudgetTests(TestCase):
    def test_lmstudio_uses_the_callers_research_budget(self) -> None:
        response = MagicMock()
        response.json.return_value = {"output": [{"type": "message", "content": "Report"}]}

        with patch("Backend.LLMProvider.requests.post", return_value=response) as post:
            provider.lmstudio_generate("Research prompt", max_output_tokens=2200)

        self.assertEqual(post.call_args.kwargs["json"]["max_output_tokens"], 2200)

    def test_openrouter_uses_the_current_completion_budget_field(self) -> None:
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "Report"}}]}

        with patch("Backend.LLMProvider.requests.post", return_value=response) as post:
            provider.openrouter_generate("Research prompt", max_output_tokens=2200)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 2200)
        self.assertNotIn("max_tokens", payload)

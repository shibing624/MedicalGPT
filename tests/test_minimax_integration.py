# -*- coding: utf-8 -*-
"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Skip them in CI environments without the key.
"""

import os
import sys
import unittest

# Add role_play_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'role_play_data'))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_create_client_and_complete(self):
        """Test creating a MiniMax client and making a real API call."""
        from llm_client import create_llm_client

        client, model = create_llm_client(provider="minimax")
        self.assertEqual(model, "MiniMax-M2.7")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            temperature=0.1,
        )
        self.assertIsNotNone(response.choices)
        self.assertTrue(len(response.choices) > 0)
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_minimax_chinese_completion(self):
        """Test MiniMax can generate Chinese medical dialogue."""
        from llm_client import create_llm_client

        client, model = create_llm_client(provider="minimax")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一名专业的护士。"},
                {"role": "user", "content": "你好，我最近头疼得厉害。请用一句话回复。"},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_generate_function(self):
        """Test the generate() helper function with real API."""
        from llm_client import create_llm_client
        from roleplay_data_generate_minimax import generate

        client, model = create_llm_client(provider="minimax")

        result = generate(
            client, model,
            "你说一句话，完成本轮对话即可。患者:",
            system_prompt="你是一名焦虑的患者，正在医院看病。"
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()

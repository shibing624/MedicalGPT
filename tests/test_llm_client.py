# -*- coding: utf-8 -*-
"""Unit tests for llm_client module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add role_play_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'role_play_data'))

from llm_client import (
    PROVIDER_CONFIGS,
    detect_provider,
    create_llm_client,
)


class TestProviderConfigs(unittest.TestCase):
    """Test provider configuration constants."""

    def test_all_providers_have_required_keys(self):
        required_keys = {"env_key", "base_url", "default_model"}
        for name, config in PROVIDER_CONFIGS.items():
            self.assertEqual(set(config.keys()), required_keys, f"Provider {name} missing keys")

    def test_openai_config(self):
        cfg = PROVIDER_CONFIGS["openai"]
        self.assertEqual(cfg["env_key"], "OPENAI_API_KEY")
        self.assertIsNone(cfg["base_url"])
        self.assertEqual(cfg["default_model"], "gpt-4o")

    def test_minimax_config(self):
        cfg = PROVIDER_CONFIGS["minimax"]
        self.assertEqual(cfg["env_key"], "MINIMAX_API_KEY")
        self.assertEqual(cfg["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(cfg["default_model"], "MiniMax-M2.7")

    def test_doubao_config(self):
        cfg = PROVIDER_CONFIGS["doubao"]
        self.assertEqual(cfg["env_key"], "DOUBAO_API_KEY")
        self.assertIn("ark.cn-beijing.volces.com", cfg["base_url"])

    def test_minimax_base_url_uses_correct_domain(self):
        """MiniMax API should use api.minimax.io, not api.minimax.chat."""
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["base_url"], "https://api.minimax.io/v1")


class TestDetectProvider(unittest.TestCase):
    """Test auto-detection of LLM providers."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_returns_none(self):
        self.assertIsNone(detect_provider())

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_detect_openai(self):
        self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True)
    def test_detect_minimax(self):
        self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"DOUBAO_API_KEY": "db-test"}, clear=True)
    def test_detect_doubao(self):
        self.assertEqual(detect_provider(), "doubao")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "MINIMAX_API_KEY": "mm-test"}, clear=True)
    def test_openai_takes_precedence(self):
        """OpenAI should be detected first when multiple keys are present."""
        self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test", "DOUBAO_API_KEY": "db-test"}, clear=True)
    def test_minimax_before_doubao(self):
        """MiniMax should be detected before Doubao."""
        self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True)
    def test_empty_key_ignored(self):
        """Empty string env vars should not count as set."""
        self.assertIsNone(detect_provider())


class TestCreateLlmClient(unittest.TestCase):
    """Test create_llm_client factory function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_provider_no_env_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client()
        self.assertIn("No LLM provider detected", str(ctx.exception))

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client(provider="unknown_provider", api_key="key")
        self.assertIn("Unknown provider", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client(provider="minimax")
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("llm_client.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}, clear=True)
    def test_minimax_client_creation(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(provider="minimax")

        mock_openai.assert_called_once_with(
            api_key="test-key-123",
            base_url="https://api.minimax.io/v1",
        )
        self.assertEqual(model, "MiniMax-M2.7")

    @patch("llm_client.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_openai_client_no_base_url(self, mock_openai):
        """OpenAI client should not set base_url (uses default)."""
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(provider="openai")

        mock_openai.assert_called_once_with(api_key="sk-test")
        self.assertEqual(model, "gpt-4o")

    @patch("llm_client.OpenAI")
    def test_api_key_override(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(provider="minimax", api_key="override-key")

        mock_openai.assert_called_once_with(
            api_key="override-key",
            base_url="https://api.minimax.io/v1",
        )

    @patch("llm_client.OpenAI")
    def test_model_override(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(
            provider="minimax", api_key="key", model="MiniMax-M2.5-highspeed"
        )
        self.assertEqual(model, "MiniMax-M2.5-highspeed")

    @patch("llm_client.OpenAI")
    def test_base_url_override(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(
            provider="minimax", api_key="key", base_url="https://custom.api.com/v1"
        )
        mock_openai.assert_called_once_with(
            api_key="key",
            base_url="https://custom.api.com/v1",
        )

    @patch("llm_client.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=True)
    def test_auto_detect_minimax(self, mock_openai):
        """Auto-detection should find MiniMax when only its key is set."""
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client()

        mock_openai.assert_called_once_with(
            api_key="env-key",
            base_url="https://api.minimax.io/v1",
        )
        self.assertEqual(model, "MiniMax-M2.7")

    @patch("llm_client.OpenAI")
    def test_case_insensitive_provider(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(provider="MiniMax", api_key="key")
        self.assertEqual(model, "MiniMax-M2.7")

    @patch("llm_client.OpenAI")
    @patch.dict(os.environ, {"DOUBAO_API_KEY": "db-key"}, clear=True)
    def test_doubao_client_creation(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_llm_client(provider="doubao")

        mock_openai.assert_called_once_with(
            api_key="db-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )


class TestCreateLlmClientReturnTypes(unittest.TestCase):
    """Test that create_llm_client returns correct types."""

    @patch("llm_client.OpenAI")
    def test_returns_tuple(self, mock_openai):
        mock_openai.return_value = MagicMock()
        result = create_llm_client(provider="minimax", api_key="key")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch("llm_client.OpenAI")
    def test_model_is_string(self, mock_openai):
        mock_openai.return_value = MagicMock()
        _, model = create_llm_client(provider="minimax", api_key="key")
        self.assertIsInstance(model, str)


if __name__ == "__main__":
    unittest.main()

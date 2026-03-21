# -*- coding: utf-8 -*-
"""Unit tests for roleplay_data_generate_minimax module."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add role_play_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'role_play_data'))

from roleplay_data_generate_minimax import generate


class TestGenerate(unittest.TestCase):
    """Test the generate() function."""

    def _mock_client(self, response_text="Mock response"):
        client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = response_text
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        client.chat.completions.create.return_value = mock_completion
        return client

    def test_generate_returns_string(self):
        client = self._mock_client("患者回复内容")
        result = generate(client, "MiniMax-M2.7", "test prompt")
        self.assertEqual(result, "患者回复内容")

    def test_generate_passes_model(self):
        client = self._mock_client()
        generate(client, "MiniMax-M2.5-highspeed", "prompt")
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5-highspeed")

    def test_generate_passes_system_prompt(self):
        client = self._mock_client()
        generate(client, "MiniMax-M2.7", "user msg", system_prompt="sys msg")
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "sys msg")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "user msg")

    def test_generate_default_system_prompt_is_empty(self):
        client = self._mock_client()
        generate(client, "MiniMax-M2.7", "prompt")
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["content"], "")

    def test_generate_uses_temperature(self):
        client = self._mock_client()
        generate(client, "MiniMax-M2.7", "prompt")
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.9)

    def test_generate_sets_max_tokens(self):
        client = self._mock_client()
        generate(client, "MiniMax-M2.7", "prompt")
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 3048)


class TestMainFunction(unittest.TestCase):
    """Test the main() data generation workflow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.seed_nurse = os.path.join(self.tmpdir, "seed_nurse_role.jsonl")
        self.seed_patient = os.path.join(self.tmpdir, "seed_patient_role.jsonl")
        self.output_file = os.path.join(self.tmpdir, "output.jsonl")

        # Write minimal seed data
        with open(self.seed_nurse, "w", encoding="utf-8") as f:
            f.write(json.dumps({"system_prompt": "你是一名经验丰富的护士"}, ensure_ascii=False) + "\n")
        with open(self.seed_patient, "w", encoding="utf-8") as f:
            f.write(json.dumps({"system_prompt": "你是一名焦虑的患者"}, ensure_ascii=False) + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_main_generates_conversations(self, mock_create):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "测试回复内容"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_create.return_value = (mock_client, "MiniMax-M2.7")

        from roleplay_data_generate_minimax import main

        test_args = [
            "prog",
            "--total", "1",
            "--output", self.output_file,
            "--rounds", "2",
        ]
        with patch("sys.argv", test_args):
            # Change to tmpdir so seed files are found
            old_cwd = os.getcwd()
            os.chdir(self.tmpdir)
            try:
                main()
            finally:
                os.chdir(old_cwd)

        # Verify output file
        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        conv = json.loads(lines[0])
        self.assertIn("id", conv)
        self.assertIn("system_prompt", conv)
        self.assertIn("conversations", conv)
        # 2 rounds * 2 turns (patient + nurse) = 4 conversation entries
        self.assertEqual(len(conv["conversations"]), 4)

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_main_conversation_format(self, mock_create):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "对话内容"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_create.return_value = (mock_client, "MiniMax-M2.7")

        from roleplay_data_generate_minimax import main

        test_args = [
            "prog",
            "--total", "1",
            "--output", self.output_file,
            "--rounds", "1",
        ]
        with patch("sys.argv", test_args):
            old_cwd = os.getcwd()
            os.chdir(self.tmpdir)
            try:
                main()
            finally:
                os.chdir(old_cwd)

        with open(self.output_file, "r", encoding="utf-8") as f:
            conv = json.loads(f.readline())

        # First entry should be from human (patient)
        self.assertEqual(conv["conversations"][0]["from"], "human")
        # Second entry should be from gpt (nurse)
        self.assertEqual(conv["conversations"][1]["from"], "gpt")

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_main_calls_create_llm_client_with_minimax(self, mock_create):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "回复"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_create.return_value = (mock_client, "MiniMax-M2.7")

        from roleplay_data_generate_minimax import main

        test_args = [
            "prog",
            "--total", "1",
            "--output", self.output_file,
            "--rounds", "1",
        ]
        with patch("sys.argv", test_args):
            old_cwd = os.getcwd()
            os.chdir(self.tmpdir)
            try:
                main()
            finally:
                os.chdir(old_cwd)

        mock_create.assert_called_once_with(provider="minimax", model=None)


if __name__ == "__main__":
    unittest.main()

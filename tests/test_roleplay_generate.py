# -*- coding: utf-8 -*-
"""Unit and integration tests for roleplay_data_generate_minimax module."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call

import pytest

# Add role_play_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'role_play_data'))

from roleplay_data_generate_minimax import generate, main


class TestGenerate(unittest.TestCase):
    """Test the generate() function."""

    def test_generate_sends_correct_messages(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="测试回复"))]
        )

        result = generate(mock_client, "MiniMax-M2.7", "你好", system_prompt="系统提示")

        mock_client.chat.completions.create.assert_called_once_with(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "你好"},
            ],
            max_tokens=3048,
            temperature=0.9,
        )
        self.assertEqual(result, "测试回复")

    def test_generate_without_system_prompt(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="回复"))]
        )

        result = generate(mock_client, "MiniMax-M2.7", "你好")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "")
        self.assertEqual(result, "回复")

    def test_generate_returns_string(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="string response"))]
        )

        result = generate(mock_client, "MiniMax-M2.7", "test")
        self.assertIsInstance(result, str)

    def test_generate_uses_correct_model(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        generate(mock_client, "MiniMax-M2.5-highspeed", "test")
        called_model = mock_client.chat.completions.create.call_args[1]["model"]
        self.assertEqual(called_model, "MiniMax-M2.5-highspeed")

    def test_generate_temperature_is_valid(self):
        """Temperature should be within MiniMax's accepted range."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        generate(mock_client, "MiniMax-M2.7", "test")
        temp = mock_client.chat.completions.create.call_args[1]["temperature"]
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)


class TestMainCLI(unittest.TestCase):
    """Test the main() CLI function."""

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_main_creates_minimax_client(self, mock_create):
        """main() should create a MiniMax client by default."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="回复内容"))]
        )
        mock_create.return_value = (mock_client, "MiniMax-M2.7")

        # Create temporary seed files
        with tempfile.TemporaryDirectory() as tmpdir:
            nurse_file = os.path.join(tmpdir, "seed_nurse_role.jsonl")
            patient_file = os.path.join(tmpdir, "seed_patient_role.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            with open(nurse_file, "w") as f:
                f.write(json.dumps({"system_prompt": "你是一名急诊护士"}) + "\n")
            with open(patient_file, "w") as f:
                f.write(json.dumps({"system_prompt": "你是一名感冒患者"}) + "\n")

            test_args = [
                "roleplay_data_generate_minimax.py",
                "--total", "1",
                "--rounds", "1",
                "--output", output_file,
            ]

            with patch("sys.argv", test_args), \
                 patch("builtins.open", wraps=open) as mock_open_fn, \
                 patch("os.path.exists", return_value=True):
                # Change to tmpdir so seed files are found
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    main()
                finally:
                    os.chdir(old_cwd)

            mock_create.assert_called_once_with(provider="minimax", model=None)

            # Verify output file was created
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertIn("conversations", data)
            self.assertIn("system_prompt", data)

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_main_custom_model(self, mock_create):
        """main() should pass custom model to create_llm_client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="回复"))]
        )
        mock_create.return_value = (mock_client, "MiniMax-M2.5-highspeed")

        with tempfile.TemporaryDirectory() as tmpdir:
            nurse_file = os.path.join(tmpdir, "seed_nurse_role.jsonl")
            patient_file = os.path.join(tmpdir, "seed_patient_role.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            with open(nurse_file, "w") as f:
                f.write(json.dumps({"system_prompt": "护士角色"}) + "\n")
            with open(patient_file, "w") as f:
                f.write(json.dumps({"system_prompt": "患者角色"}) + "\n")

            test_args = [
                "roleplay_data_generate_minimax.py",
                "--model", "MiniMax-M2.5-highspeed",
                "--total", "1",
                "--rounds", "1",
                "--output", output_file,
            ]

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("sys.argv", test_args):
                    main()
            finally:
                os.chdir(old_cwd)

            mock_create.assert_called_once_with(
                provider="minimax", model="MiniMax-M2.5-highspeed"
            )

    @patch("roleplay_data_generate_minimax.create_llm_client")
    def test_output_format_has_conversation_pairs(self, mock_create):
        """Each conversation should have paired human/gpt turns."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="对话内容"))]
        )
        mock_create.return_value = (mock_client, "MiniMax-M2.7")

        with tempfile.TemporaryDirectory() as tmpdir:
            nurse_file = os.path.join(tmpdir, "seed_nurse_role.jsonl")
            patient_file = os.path.join(tmpdir, "seed_patient_role.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            with open(nurse_file, "w") as f:
                f.write(json.dumps({"system_prompt": "护士"}) + "\n")
            with open(patient_file, "w") as f:
                f.write(json.dumps({"system_prompt": "患者"}) + "\n")

            test_args = [
                "roleplay_data_generate_minimax.py",
                "--total", "1",
                "--rounds", "3",
                "--output", output_file,
            ]

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("sys.argv", test_args):
                    main()
            finally:
                os.chdir(old_cwd)

            with open(output_file, "r") as f:
                data = json.loads(f.readline())

            convs = data["conversations"]
            # 3 rounds = 6 turns (3 human + 3 gpt)
            self.assertEqual(len(convs), 6)
            for i, turn in enumerate(convs):
                expected_from = "human" if i % 2 == 0 else "gpt"
                self.assertEqual(turn["from"], expected_from)


class TestIntegrationMiniMax(unittest.TestCase):
    """Integration tests that call the real MiniMax API.

    These tests are skipped unless MINIMAX_API_KEY is set in the environment.
    """

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get("MINIMAX_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("MINIMAX_API_KEY not set, skipping integration tests")
        from llm_client import create_llm_client
        cls.client, cls.model = create_llm_client(provider="minimax")

    @pytest.mark.timeout(120)
    def test_minimax_simple_generation(self):
        """Test a simple generation call to MiniMax API."""
        response = generate(
            self.client, self.model,
            prompt="请用一句话介绍自己。",
            system_prompt="你是一名护士。",
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    @pytest.mark.timeout(120)
    def test_minimax_chinese_response(self):
        """MiniMax should return Chinese text for Chinese prompts."""
        response = generate(
            self.client, self.model,
            prompt="你好，你是谁？",
            system_prompt="你是一名医院护士。",
        )
        # Check that response contains at least some CJK characters
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in response)
        self.assertTrue(has_cjk, f"Expected Chinese response, got: {response}")

    @pytest.mark.timeout(180)
    def test_minimax_generate_one_conversation(self):
        """Generate a single 2-turn conversation and verify format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nurse_file = os.path.join(tmpdir, "seed_nurse_role.jsonl")
            patient_file = os.path.join(tmpdir, "seed_patient_role.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            with open(nurse_file, "w") as f:
                f.write(json.dumps({"system_prompt": "你是一名经验丰富的急诊科护士"}) + "\n")
            with open(patient_file, "w") as f:
                f.write(json.dumps({"system_prompt": "你是一名感冒发烧的年轻患者"}) + "\n")

            test_args = [
                "roleplay_data_generate_minimax.py",
                "--total", "1",
                "--rounds", "2",
                "--output", output_file,
            ]

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("sys.argv", test_args):
                    main()
            finally:
                os.chdir(old_cwd)

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r") as f:
                data = json.loads(f.readline())

            self.assertIn("conversations", data)
            self.assertIn("system_prompt", data)
            self.assertEqual(len(data["conversations"]), 4)  # 2 rounds * 2 turns
            self.assertEqual(data["conversations"][0]["from"], "human")
            self.assertEqual(data["conversations"][1]["from"], "gpt")


if __name__ == "__main__":
    unittest.main()

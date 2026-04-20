import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.opd_training import OPDScriptArguments, convert_sharegpt_batch_to_messages


def test_opd_helpers_are_inlined_into_opd_training_module():
    assert not Path(__file__).resolve().parents[1].joinpath("training", "opd_utils.py").exists()


def test_convert_sharegpt_batch_to_messages_builds_multi_turn_samples():
    examples = {
        "conversations": [[
            {"from": "system", "value": "You are a helpful doctor."},
            {"from": "human", "value": "What is a fever?"},
            {"from": "gpt", "value": "A fever is an elevated body temperature."},
            {"from": "human", "value": "When should I see a doctor?"},
            {"from": "gpt", "value": "Seek care if the fever is persistent or severe."},
        ]]
    }

    result = convert_sharegpt_batch_to_messages(examples)

    assert len(result["messages"]) == 2
    assert result["messages"][0] == [
        {"role": "system", "content": "You are a helpful doctor."},
        {"role": "user", "content": "What is a fever?"},
        {"role": "assistant", "content": "A fever is an elevated body temperature."},
    ]
    assert result["messages"][1] == [
        {"role": "system", "content": "You are a helpful doctor."},
        {"role": "user", "content": "What is a fever?"},
        {"role": "assistant", "content": "A fever is an elevated body temperature."},
        {"role": "user", "content": "When should I see a doctor?"},
        {"role": "assistant", "content": "Seek care if the fever is persistent or severe."},
    ]


def test_convert_sharegpt_batch_to_messages_formats_tool_calls_and_observations():
    examples = {
        "conversations": [[
            {"from": "human", "value": "Check Beijing weather"},
            {"from": "function_call", "value": '{"name": "get_weather", "arguments": {"city": "Beijing"}}'},
            {"from": "observation", "value": '{"temperature": "25C"}'},
            {"from": "gpt", "value": "It is 25C in Beijing."},
        ]],
        "tools": ['[{"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]'],
    }

    result = convert_sharegpt_batch_to_messages(examples, tool_format="default")

    assert len(result["messages"]) == 2
    assert result["messages"][0][0]["role"] == "system"
    assert "You have access to the following tools" in result["messages"][0][0]["content"]
    assert result["messages"][0][1]["role"] == "user"
    assert result["messages"][0][2]["role"] == "assistant"
    assert "Action: get_weather" in result["messages"][0][2]["content"]
    assert result["messages"][1][3] == {
        "role": "user",
        "content": 'Observation: {"temperature": "25C"}',
    }
    assert result["messages"][1][4] == {
        "role": "assistant",
        "content": "It is 25C in Beijing.",
    }


def test_opd_script_arguments_validate_ranges():
    with pytest.raises(ValueError, match="opd_lambda"):
        OPDScriptArguments(opd_lambda=1.5)

    with pytest.raises(ValueError, match="opd_beta"):
        OPDScriptArguments(opd_beta=-0.1)

    with pytest.raises(ValueError, match="max_prompt_length"):
        OPDScriptArguments(max_prompt_length=0)

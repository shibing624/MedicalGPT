import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.tool_utils import load_local_json_datasets


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_local_json_datasets_aligns_optional_columns(tmp_path):
    regular_file = tmp_path / "regular.jsonl"
    tool_file = tmp_path / "tool.jsonl"

    _write_jsonl(
        regular_file,
        [
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "world"},
                ]
            }
        ],
    )
    _write_jsonl(
        tool_file,
        [
            {
                "conversations": [
                    {"from": "human", "value": "weather"},
                    {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {}}"},
                    {"from": "observation", "value": "{\"temperature\": \"25C\"}"},
                    {"from": "gpt", "value": "done"},
                ],
                "tools": "[]",
            }
        ],
    )

    dataset_dict = load_local_json_datasets({"train": [str(regular_file), str(tool_file)]})
    train_dataset = dataset_dict["train"]

    assert len(train_dataset) == 2
    assert "tools" in train_dataset.column_names
    assert train_dataset[0]["tools"] is None
    assert train_dataset[1]["tools"] == "[]"


def test_load_local_json_datasets_aligns_optional_tools_for_preference_data(tmp_path):
    regular_file = tmp_path / "regular_preference.jsonl"
    tool_file = tmp_path / "tool_preference.jsonl"

    _write_jsonl(
        regular_file,
        [
            {
                "conversations": [{"from": "human", "value": "question"}],
                "chosen": "better answer",
                "rejected": "worse answer",
            }
        ],
    )
    _write_jsonl(
        tool_file,
        [
            {
                "conversations": [{"from": "human", "value": "need tool"}],
                "tools": "[]",
                "chosen": "Action: search\nAction Input: {}",
                "rejected": "No tool",
            }
        ],
    )

    dataset_dict = load_local_json_datasets({"train": [str(regular_file), str(tool_file)]})
    train_dataset = dataset_dict["train"]

    assert len(train_dataset) == 2
    assert "tools" in train_dataset.column_names
    assert train_dataset[0]["tools"] is None
    assert train_dataset[1]["tools"] == "[]"
    assert train_dataset[0]["chosen"] == "better answer"
    assert train_dataset[1]["rejected"] == "No tool"

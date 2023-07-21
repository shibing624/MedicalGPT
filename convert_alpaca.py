"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_alpaca.py --in alpaca_data.json
"""

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r", encoding="utf-8"))
    new_content = []
    for i, c in enumerate(content):
        if len(c["input"].strip()) > 1:
            q, a = c["instruction"] + "\nInput:\n" + c["input"], c["output"]
        else:
            q, a = c["instruction"], c["output"]
        new_content.append(
            {
                "id": f"alpaca_{i}",
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ],
            }
        )

    print(f"#out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Preprocess the Numia dataset to ShareGPT format and save as JSONL
"""

import os
import json
import argparse
import datasets


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
        else:
            return None

    left = "\\boxed{"

    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    else:
        return None


def last_boxed_only_string(string):
    if string is None:
        return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    try:
        boxed_string = last_boxed_only_string(solution_str)
        if boxed_string is None:
            return None
        return remove_boxed(boxed_string)
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/numina_cot')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=1000)
    parser.add_argument('--output_file', default='numina_cot_sharegpt_data.jsonl')

    args = parser.parse_args()

    data_source = 'AI-MO/NuminaMath-CoT'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))
    print(f"Loaded {len(train_dataset)} training examples from {data_source} dataset.", flush=True)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Convert to ShareGPT format
    def make_sharegpt_format(example, idx):
        question_raw = example.pop('problem')
        question = question_raw + ' ' + instruction_following

        answer_raw = example.pop('solution')
        solution = answer_raw # extract_solution(answer_raw)

        return {
            "id": f"{data_source}-{idx}",
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": solution if solution else "No solution available."}
            ]
        }

    # Map dataset to ShareGPT format
    sharegpt_data = train_dataset.map(
        lambda example, idx: make_sharegpt_format(example, idx),
        with_indices=True, remove_columns=train_dataset.column_names
    )

    # Save to JSONL
    output_file = os.path.join(args.local_dir, args.output_file)
    os.makedirs(args.local_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sharegpt_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(sharegpt_data)} ShareGPT formatted examples to {output_file}")
"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_alpaca.py --in_file alpaca_data.json --out_file alpaca_data_sharegpt.json
"""

import argparse

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    args = parser.parse_args()

    data_files = {"train": args.in_file}
    raw_datasets = load_dataset('json', data_files=data_files)

    def process(examples):
        ids = []
        convs = []
        id = 0
        for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
            if len(inp.strip) > 1:
                instruction = instruction + '\nInput:\n' + inp
            q = instruction
            a = output
            convs.append([
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ])
            id += 1
            ids.append(f'alpaca_{id}')
        return {'id': ids, 'conversations': convs}


    dataset = raw_datasets.map(process, batched=True, remove_columns=raw_datasets['train'].column_names)
    dataset.to_json(f"{args.out_file}", lines=True, force_ascii=False)

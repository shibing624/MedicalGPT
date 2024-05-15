"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_dataset.py --in_file alpaca_data.json --out_file alpaca_data_sharegpt.jsonl
"""

import argparse

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--data_type", type=str, default='alpaca', help="alpaca, qa, or sharegpt")
    parser.add_argument("--file_type", type=str, default='json')
    args = parser.parse_args()
    print(args)
    data_files = {"train": args.in_file}
    if args.file_type == 'csv':
        if args.data_type in ['qa']:
            column_names = ['input', 'output']
        else:
            column_names = ['instruction', 'input', 'output']
        raw_datasets = load_dataset('csv', data_files=data_files, column_names=column_names, delimiter='\t')
    elif args.file_type in ['json', 'jsonl']:
        raw_datasets = load_dataset('json', data_files=data_files)
    else:
        raise ValueError("File type not supported")
    ds = raw_datasets['train']


    def process_qa(examples):
        convs = []
        for q, a in zip(examples['input'], examples['output']):
            convs.append([
                {"from": "human", "value": q},
                {"from": "gpt", "value": a}
            ])
        return {"conversations": convs}


    def process_alpaca(examples):
        convs = []
        for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
            if inp and len(inp.strip()) > 0:
                instruction = instruction + '\n\n' + inp
            q = instruction
            a = output
            convs.append([
                {"from": "human", "value": q},
                {"from": "gpt", "value": a}
            ])
        return {"conversations": convs}


    if args.data_type in ['alpaca']:
        ds = ds.map(process_alpaca, batched=True, remove_columns=ds.column_names, desc="Running process")
    elif args.data_type in ['qa']:
        ds = ds.map(process_qa, batched=True, remove_columns=ds.column_names, desc="Running process")
    else:
        # Other sharegpt dataset, need rename to conversations and remove unused columns
        if "items" in ds.column_names:
            ds = ds.rename(columns={"items": "conversations"})
        columns_to_remove = ds.column_names.copy()
        columns_to_remove.remove('conversations')
        ds = ds.remove_columns(columns_to_remove)

    ds.to_json(f"{args.out_file}", lines=True, force_ascii=False)

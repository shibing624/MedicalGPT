"""
Convert dataset formats to ShareGPT jsonl.

Supported conversions:
  alpaca  -> sharegpt jsonl:  {instruction, input, output} -> {conversations: [{from, value}]}
  qa      -> sharegpt jsonl:  {input, output}              -> {conversations: [{from, value}]}
  json    -> jsonl:           JSON array file               -> one JSON object per line

Usage:
  python convert_dataset.py --in_file alpaca_data.json --out_file out.jsonl --data_type alpaca
  python convert_dataset.py --in_file qa_data.csv --out_file out.jsonl --data_type qa --file_type csv
  python convert_dataset.py --in_file data.json --out_file data.jsonl --data_type json2jsonl
"""

import argparse
import json

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True, help="Input file name.")
    parser.add_argument("--out_file", type=str, required=True, help="Output file name, e.g. out.jsonl")
    parser.add_argument("--data_type", type=str, default='alpaca',
                        help="alpaca, qa, json2jsonl, or sharegpt")
    parser.add_argument("--file_type", type=str, default='json', help='Input file type: json or csv')
    args = parser.parse_args()
    print(args)

    if args.data_type == 'json2jsonl':
        with open(args.in_file) as f:
            data = json.load(f)
        with open(args.out_file, 'w') as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        print(f"Converted {len(data)} samples: {args.in_file} -> {args.out_file}")
    else:
        data_files = {"train": args.in_file}
        if args.file_type == 'csv':
            if args.data_type == 'qa':
                column_names = ['input', 'output']
            else:
                column_names = ['instruction', 'input', 'output']
            raw_datasets = load_dataset('csv', data_files=data_files, column_names=column_names, delimiter='\t')
        elif args.file_type in ['json', 'jsonl']:
            raw_datasets = load_dataset('json', data_files=data_files)
        else:
            raise ValueError(f"File type not supported: {args.file_type}")
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
                convs.append([
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": output}
                ])
            return {"conversations": convs}

        if args.data_type == 'alpaca':
            ds = ds.map(process_alpaca, batched=True, remove_columns=ds.column_names, desc="Running process")
        elif args.data_type == 'qa':
            ds = ds.map(process_qa, batched=True, remove_columns=ds.column_names, desc="Running process")
        else:
            if "items" in ds.column_names:
                ds = ds.rename(columns={"items": "conversations"})
            columns_to_remove = ds.column_names.copy()
            columns_to_remove.remove('conversations')
            ds = ds.remove_columns(columns_to_remove)

        ds.to_json(f"{args.out_file}", lines=True, force_ascii=False)
        print(f"Converted {len(ds)} samples: {args.in_file} -> {args.out_file}")

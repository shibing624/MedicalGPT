# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Usage:
python merge_peft_adapter.py \
    --base_model_name_or_path path/to/llama/model \
    --peft_model_path path/to/first/lora/model \
    --output_dir path/to/output/dir
"""

import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name_or_path', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--peft_model_path', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--output_dir', default='./merged', type=str)

    args = parser.parse_args()
    print(args)
    base_model_path = args.base_model_name_or_path
    peft_model_path = args.peft_model_path
    output_dir = args.output_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {peft_model_path}")
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    if peft_config.task_type == "SEQ_CLS":
        print("Loading LoRA for sequence classification model")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print("Loading LoRA for causal language model")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Extended vocabulary size to {len(tokenizer)}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = lora_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir)
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()

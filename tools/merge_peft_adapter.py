# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Usage:
python merge_peft_adapter.py \
    --base_model path/to/llama/model \
    --tokenizer_path path/to/llama/tokenizer \
    --lora_model path/to/lora/model \
    --output_dir path/to/output/dir
"""

import argparse
import glob
import os
import shutil

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForConditionalGeneration,
    AutoModelForSequenceClassification,
)


def _overwrite_tokenizer_files_from_base(base_dir: str, output_dir: str):
    """从 base model 目录复制 tokenizer 相关文件到 output_dir，覆盖 save_pretrained 的输出。

    transformers 的 save_pretrained 可能丢失关键字段（如 chat_template、
    added_tokens_decoder、preprocessor_config 等），直接从原始目录覆盖更可靠。
    """
    # 需要从 base model 复制的文件（如果存在的话）
    files_to_copy = [
        'tokenizer_config.json',
        'tokenizer.json',
        'chat_template.jinja',
        'preprocessor_config.json',
        'video_preprocessor_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
        'configuration.json',
        'generation_config.json',
    ]
    # 也复制 tokenizer.model 等二进制文件
    for pattern in ['tokenizer.model', 'tokenizer.model.*']:
        for f in glob.glob(os.path.join(base_dir, pattern)):
            files_to_copy.append(os.path.basename(f))

    copied = []
    for fname in files_to_copy:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied.append(fname)
    if copied:
        print(f"Copied tokenizer/config files from {base_dir}: {copied}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Please specify tokenization path.")
    parser.add_argument('--lora_model', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--output_dir', default='./merged', type=str)
    parser.add_argument('--hf_hub_model_id', default='', type=str)
    parser.add_argument('--hf_hub_token', default=None, type=str)
    args = parser.parse_args()
    print(args)

    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    peft_config = PeftConfig.from_pretrained(lora_model_path)

    if peft_config.task_type == "SEQ_CLS":
        print("Loading LoRA for sequence classification model")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        # 自动判断模型架构：如果是多模态模型（如 Qwen3_5ForConditionalGeneration），
        # 用 AutoModelForConditionalGeneration 加载以保留原始 config 结构；
        # 否则用 AutoModelForCausalLM
        base_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        archs = getattr(base_cfg, 'architectures', []) or []
        is_conditional = any('ConditionalGeneration' in a for a in archs)
        if is_conditional:
            print(f"Loading LoRA for conditional generation model (archs={archs})")
            base_model = AutoModelForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype='auto',
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            print(f"Loading LoRA for causal language model (archs={archs})")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype='auto',
                trust_remote_code=True,
                device_map="auto",
            )
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if args.resize_emb:
        base_model_token_size = base_model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype='auto',
    )
    new_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = new_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir, max_shard_size='10GB')

    # 从 base model 目录补全 tokenizer 相关文件，避免 save_pretrained 丢失关键字段
    # （如 chat_template、added_tokens_decoder、preprocessor_config.json 等）
    tokenizer_src = args.tokenizer_path or base_model_path
    _overwrite_tokenizer_files_from_base(tokenizer_src, output_dir)

    print(f"Done! model saved to {output_dir}")
    if args.hf_hub_model_id:
        print(f"Pushing to Hugging Face Hub...")
        base_model.push_to_hub(
            args.hf_hub_model_id,
            token=args.hf_hub_token,
            max_shard_size="10GB",
        )
        tokenizer.push_to_hub(
            args.hf_hub_model_id,
            token=args.hf_hub_token,
        )
        print(f"Done! model pushed to Hugging Face Hub: {args.hf_hub_model_id}")


if __name__ == '__main__':
    main()

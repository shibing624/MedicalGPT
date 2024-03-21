# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Usage:
python merge_peft_adapter.py \
    --model_type llama \
    --base_model path/to/llama/model \
    --tokenizer_path path/to/llama/tokenizer \
    --lora_model path/to/lora/model \
    --output_dir path/to/output/dir
"""

import argparse
import os
from algorithm.llm.utils.EnvUtils import use_modelscope
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}
def get_chinese_download(path,cache_dir):
    try:
        from modelscope import snapshot_download
        # revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        return  snapshot_download(
            path, revision="master", cache_dir=cache_dir
        )
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Please specify tokenization path.")
    parser.add_argument('--lora_model', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--output_dir', default='./merged', type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    args = parser.parse_args()
    print(args)

    base_model_path = args.base_model
    lora_model_path = args.lora_model
    if not use_modelscope() or os.path.exists(base_model_path):
        pass
    else:
        base_model_path = get_chinese_download(base_model_path,cache_dir=args.cache_dir)

    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    peft_config = PeftConfig.from_pretrained(lora_model_path)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if peft_config.task_type == "SEQ_CLS":
        print("Loading LoRA for sequence classification model")
        if args.model_type == "chatglm":
            raise ValueError("chatglm does not support sequence classification")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        print("Loading LoRA for causal language model")
        base_model = model_class.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    if args.tokenizer_path:
        if not use_modelscope() or os.path.exists(args.tokenizer_path):
            pass
        else:
            args.tokenizer_path = get_chinese_download(args.tokenizer_path, cache_dir=args.cache_dir)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = tokenizer_class.from_pretrained(base_model_path, trust_remote_code=True)
    if args.resize_emb:
        base_model_token_size = base_model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    new_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = new_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir, safe_serialization=False)  # max_shard_size='10GB'
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()

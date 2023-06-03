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

import peft
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name_or_path', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--peft_model_path', default=None, required=True,
                        type=str,
                        help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
    parser.add_argument('--offload_dir', default=None, type=str,
                        help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
    parser.add_argument('--output_dir', default='./merged', type=str)

    args = parser.parse_args()
    print(args)
    base_model_path = args.base_model_name_or_path
    lora_model_paths = [s.strip() for s in args.peft_model_path.split(',') if s.strip()]
    output_dir = args.output_dir
    offload_dir = args.offload_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    if offload_dir is not None:
        # Load with offloading, which is useful for low-RAM machines.
        # Note that if you have enough RAM, please use original method instead, as it is faster.
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
    else:
        # Original method without offloading
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Infer the model size from the checkpoint
    embedding_size = base_model.get_input_embeddings().weight.size(1)
    emb_to_model_size = {
        4096: '7B',
        5120: '13B',
        6656: '30B',
        8192: '65B',
    }
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    tokenizer = None
    for lora_index, lora_model_path in enumerate(lora_model_paths):
        print(f"Loading LoRA {lora_model_path}")
        tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
        if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Extended vocabulary size to {len(tokenizer)}")

        first_weight = base_model.model.layers[0].self_attn.q_proj.weight
        first_weight_old = first_weight.clone()

        if hasattr(peft.LoraModel, 'merge_and_unload'):
            lora_model = PeftModel.from_pretrained(
                base_model,
                lora_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            assert torch.allclose(first_weight_old, first_weight)
            print(f"Merging with merge_and_unload...")
            base_model = lora_model.merge_and_unload()

    tokenizer.save_pretrained(output_dir)
    print("Saving to Hugging Face format...")
    LlamaForCausalLM.save_pretrained(base_model, output_dir)
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()

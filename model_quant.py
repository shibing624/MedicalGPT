# -*- coding: utf-8 -*-
"""
@author:ZhuangXialie(1832963123@qq.com)
@description: model quantify

usage:
python model_quant.py --unquantized_model_path /path/to/unquantized/model --quantized_model_output_path /path/to/save/quantized/model --input_text "Your input text here"
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="量化模型推理对比")
    parser.add_argument("--unquantized_model_path", type=str, required=True, help="未量化模型路径")
    parser.add_argument("--quantized_model_output_path", type=str, required=True, help="量化模型保存路径")
    parser.add_argument("--input_text", type=str, required=True, help="输入的文本内容")
    return parser.parse_args()


# 计算模型相关的显存占用
def get_model_memory_usage(device):
    return torch.cuda.memory_allocated(device) / (1024 ** 3)  # 转换为GB


# 定义一个函数来进行推理，并计算推理时间
def perform_inference(model, tokenizer, devic, question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = inputs["attention_mask"]

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id 为 eos_token_id
        )
    end_time = time.time()
    elapsed_time = end_time - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, elapsed_time


def main():
    args = parse_args()

    # 1. 未量化模型推理和显存计算
    print("\n====== 未量化模型推理 ======")
    tokenizer = AutoTokenizer.from_pretrained(args.unquantized_model_path, trust_remote_code=True)

    gpu_memory_before_unquantized = get_model_memory_usage(device)  # 模型加载前的显存
    unquantized_model = AutoModelForCausalLM.from_pretrained(args.unquantized_model_path, trust_remote_code=True)
    unquantized_model.to(device)
    gpu_memory_after_unquantized = get_model_memory_usage(device)  # 模型加载后的显存
    model_memory_unquantized = gpu_memory_after_unquantized - gpu_memory_before_unquantized  # 计算模型显存占用
    print(f"未量化模型加载显存占用: {model_memory_unquantized:.2f} GB")

    generated_text_unquantized, time_unquantized = perform_inference(unquantized_model, tokenizer, device,
                                                                     args.input_text)
    print(f"推理生成的文本（未量化模型）: {generated_text_unquantized}")
    print(f"推理时间（未量化模型）: {time_unquantized:.2f} 秒")

    # 卸载未量化模型以释放显存
    del unquantized_model
    torch.cuda.empty_cache()

    # 重新计算显存基线
    print("\n清理缓存，重新计算显存...")
    time.sleep(2)  # 确保显存释放，等待一段时间
    gpu_memory_after_cache_clear = get_model_memory_usage(device)
    print(f"显存清理后基线显存: {gpu_memory_after_cache_clear:.2f} GB")

    # 2. 量化模型推理和保存
    print("\n====== 量化模型推理和保存 ======")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 开启4bit量化
        load_in_8bit=False,  # 禁止8bit量化
        bnb_4bit_compute_dtype=torch.float16,  # 计算数据类型为float16
        bnb_4bit_quant_storage=torch.uint8,  # 存储数据类型为uint8
        bnb_4bit_quant_type="nf4",  # 使用nf4量化类型
        bnb_4bit_use_double_quant=True  # 开启双重量化以优化推理
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        args.unquantized_model_path,
        device_map="auto",  # 自动分配设备
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    generated_text_quantized, time_quantized = perform_inference(quantized_model, tokenizer, device, args.input_text)
    print(f"推理生成的文本（量化模型）: {generated_text_quantized}")
    print(f"推理时间（量化模型）: {time_quantized:.2f} 秒")

    # 保存量化模型和tokenizer
    quantized_model.save_pretrained(args.quantized_model_output_path)
    tokenizer.save_pretrained(args.quantized_model_output_path)
    print(f"量化模型和tokenizer已保存到 {args.quantized_model_output_path}")

    # 输出对比
    print("\n====== 内容对比结果 ======")
    print(f"未量化模型生成文本:\n {generated_text_unquantized}")
    print(f"量化模型生成文本:\n {generated_text_quantized}")

    print("\n====== 时间对比结果 ======")
    print(f"未量化模型推理时间: {time_unquantized:.2f} 秒")
    print(f"量化模型推理时间: {time_quantized:.2f} 秒")


if __name__ == "__main__":
    main()

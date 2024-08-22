# -*- coding: utf-8 -*-
"""
@description: eval quantize for jsonl format data

usage:
python eval_quantize.py --bnb_path /path/to/your/bnb_model --data_path data/finetune/medical_sft_1K_format.jsonl
"""
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import GPUtil
import argparse
import logging
import os

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建参数解析器
parser = argparse.ArgumentParser(description="========量化困惑度测试========")
parser.add_argument(
    "--bnb_path",  
    type=str,  
    required=True,  # 设置为必须的参数
    help="bnb量化后的模型路径。"  
)
parser.add_argument(
    "--data_path",  
    type=str,  
    required=True,  # 设置为必须的参数
    help="jsonl数据集路径。"  
)

# 设备选择函数
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

# 清理GPU缓存函数
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 从jsonl文件中加载数据
def load_jsonl_data(file_path):
    logger.info(f"Loading data from {file_path}")
    conversations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 human 和 gpt 部分的文本
                for conv in data['conversations']:
                    if conv['from'] == 'human':
                        input_text = conv['value']
                    elif conv['from'] == 'gpt':
                        target_text = conv['value']
                        conversations.append((input_text, target_text))
        return conversations
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

# 困惑度评估函数
def evaluate_perplexity(model, tokenizer, conversation_pairs):
    def _perplexity(nlls, n_samples, seqlen):
        try:
            return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    model = model.eval()
    nlls = []

    # 遍历每个对话，基于 human 部分生成并与 gpt 部分计算困惑度
    for input_text, target_text in tqdm(conversation_pairs, desc="Perplexity Evaluation"):
        # Tokenize input and target
        inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.to(get_device())
        target_ids = tokenizer(target_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.to(get_device())

        # Ensure both inputs and target have the same length
        if inputs.size(1) != target_ids.size(1):
            logger.warning(f"Input length {inputs.size(1)} and Target length {target_ids.size(1)} are not equal.")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=target_ids)
            loss = outputs.loss
            nlls.append(loss * target_ids.size(1))  # loss * sequence length



    # 计算最终困惑度
    total_samples = len(conversation_pairs)
    total_length = sum([len(pair[1]) for pair in conversation_pairs])
    ppl = _perplexity(nlls, total_samples, total_length)
    logger.info(f"Final Perplexity: {ppl:.3f}")

    return ppl.item()

# 主函数
if __name__ == "__main__":
    
    args = parser.parse_args()

    if not os.path.exists(args.bnb_path):
        logger.error(f"Model path {args.bnb_path} does not exist.")
        exit(1)

    try:
        # 设置BNB量化配置
        from accelerate.utils import BnbQuantizationConfig
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        
        logger.info(f"Loading BNB model from: {args.bnb_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.bnb_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.bnb_path, trust_remote_code=True)
        
        # 检查GPU使用情况
        if torch.cuda.is_available():
            gpu_usage = GPUtil.getGPUs()[0].memoryUsed
            logger.info(f"GPU usage before evaluation: {round(gpu_usage/1024, 2)} GB")
        
        # 加载jsonl数据
        conversation_pairs = load_jsonl_data(args.data_path)
        
        if not conversation_pairs:
            logger.error("No valid conversation pairs found.")
            exit(1)

        # 开始评估
        evaluate_perplexity(model, tokenizer, conversation_pairs)
        
        # 评估完毕，清理模型和缓存
        del model
        clear_gpu_cache()
        logger.info("Evaluation completed and GPU cache cleared.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

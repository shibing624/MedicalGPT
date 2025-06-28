# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Accelerate SFT训练脚本
"""

import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Literal, Optional, Tuple

import torch
import torch.utils.data
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import LabelSmoother
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False
from template import get_conv_template


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: Optional[str] = field(default="main")
    hf_hub_token: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(default="float16")
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None,
                                        metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={
        "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file_dir: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file_dir: str = field(default=None, metadata={"help": "Path to the validation data."})
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    train_on_inputs: bool = field(default=False)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    model_max_length: int = field(default=2048)
    template_name: Optional[str] = field(default="vicuna")
    # 添加参数控制是否使用张量并行
    use_tensor_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to use tensor parallelism for large models"}
    )


def find_all_linear_names(model, int4=False, int8=False):
    """查找模型中所有的线性层名称"""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def save_model(model, tokenizer, output_dir):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_datasets(data_args, model_args):
    """Load datasets from files or HuggingFace hub"""
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # Loading a dataset from local files.
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]

    logger.info(f"Raw datasets: {raw_datasets}")
    return raw_datasets


def create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """Create preprocessing function for datasets"""
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                system_prompt = ""
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role == "system":
                    # Skip the first one if it is from system
                    system_prompt = source[0]["value"]
                    source = source[1:]
                    data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs:
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    return preprocess_function


def filter_empty_labels(example, IGNORE_INDEX):
    """Remove empty labels dataset."""
    return not all(label == IGNORE_INDEX for label in example["labels"])


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

        if free < 2.0:  # 如果可用内存少于2GB
            logger.warning(f"⚠️ GPU {i} 可用内存不足 ({free:.1f}GB)，建议:")
            logger.warning("  1. 使用 --load_in_4bit 启用4bit量化")
            logger.warning("  2. 减小 --per_device_train_batch_size")
            logger.warning("  3. 增加 --gradient_accumulation_steps")
            logger.warning("  4. 减小 --model_max_length")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def get_unwrapped_model(model):
    """获取未包装的原始模型，无论它是否被DDP包装"""
    if hasattr(model, "module"):
        return model.module
    return model


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    # 设置日志 - 只在主进程输出
    logger.info(f"🚀 使用Accelerate库进行多GPU训练")
    logger.info("🚀 开始初始化Accelerator...")
    # 直接创建Accelerator，让它自己处理状态
    accelerator = Accelerator()
    logger.info("✅ Accelerator初始化完成")
    try:
        logger.info(f"设备: {accelerator.device}")
        logger.info(f"检测到 {accelerator.num_processes} 个进程")
        logger.info(f"当前进程: {accelerator.process_index}")
        logger.info(f"分布式类型: {accelerator.distributed_type}")
    except:
        logger.warning("无法获取完整的Accelerator信息，但这不影响训练")

    logger.info(f"Model args: {model_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")

    # 设置随机种子
    accelerate_set_seed(training_args.seed)

    # 加载tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 设置特殊token
    prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}")

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}")

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    logger.info("✅ Tokenizer配置完成")

    # 检查和优化内存
    check_and_optimize_memory()

    logger.info("🔄 开始加载模型...")

    # 加载模型配置
    torch_dtype = model_args.torch_dtype
    # 配置量化
    quantization_config = None
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "hf_hub_token": model_args.hf_hub_token,
    }
    if model_args.flash_attn:
        if is_flash_attn_2_available:
            config_kwargs["use_flash_attention_2"] = True
            logger.info("Using FlashAttention-2 for faster training and inference.")
        else:
            logger.warning("FlashAttention-2 is not installed.")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # 检测GPU使用情况并优化内存配置
    total_memory = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"检测到 {num_gpus} 个GPU")

        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            free = gpu_memory - allocated
            total_memory += gpu_memory
            logger.info(
                f"GPU {i}: 总内存={gpu_memory:.1f}GB, 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 可用={free:.1f}GB")

        logger.info(f"总GPU内存: {total_memory:.1f}GB")

        # 清理GPU缓存
        torch.cuda.empty_cache()
        logger.info("已清理GPU缓存")

    # 估算模型大小（粗略估算）
    estimated_model_size_gb = 0
    if hasattr(config, 'num_parameters'):
        # 如果配置中有参数数量信息
        estimated_model_size_gb = config.num_parameters * 2 / 1024 ** 3  # 假设fp16
    else:
        # 根据模型名称粗略估算
        model_name_lower = model_args.model_name_or_path.lower()
        if '70b' in model_name_lower or '72b' in model_name_lower:
            estimated_model_size_gb = 140  # 70B模型大约140GB
        elif '32b' in model_name_lower or '34b' in model_name_lower:
            estimated_model_size_gb = 64  # 32B模型大约64GB
        elif '13b' in model_name_lower or '14b' in model_name_lower:
            estimated_model_size_gb = 26  # 13B模型大约26GB
        elif '7b' in model_name_lower or '8b' in model_name_lower:
            estimated_model_size_gb = 14  # 7B模型大约14GB
        elif '3b' in model_name_lower:
            estimated_model_size_gb = 6  # 3B模型大约6GB
        else:
            estimated_model_size_gb = 10  # 默认估算

    logger.info(f"估算模型大小: {estimated_model_size_gb:.1f}GB")

    # 根据模型大小和GPU数量以及用户选择决定使用DDP还是张量并行
    num_gpus = torch.cuda.device_count()
    is_distributed = accelerator.num_processes > 1

    # 智能选择加载策略
    if is_distributed:
        if script_args.use_tensor_parallel and estimated_model_size_gb > 20:
            # 用户选择使用张量并行且模型足够大
            logger.info(f"🔧 使用张量并行策略 (模型大小: {estimated_model_size_gb:.1f}GB)")
            use_tensor_parallel = True

            # 检查PyTorch版本是否支持张量并行
            import pkg_resources
            torch_version = pkg_resources.get_distribution("torch").version
            if pkg_resources.parse_version(torch_version) < pkg_resources.parse_version("2.5.0"):
                logger.warning(f"⚠️ 当前PyTorch版本 {torch_version} 不支持张量并行，需要 >= 2.5.0")
                logger.warning("⚠️ 自动切换到DDP模式")
                use_tensor_parallel = False
            else:
                logger.info(f"✅ PyTorch版本 {torch_version} 支持张量并行")
        else:
            # 使用DDP
            logger.info(f"🔧 使用DDP进行多GPU训练 (模型大小: {estimated_model_size_gb:.1f}GB)")
            use_tensor_parallel = False
    else:
        # 单进程，可以使用device_map="auto"
        logger.info("🔧 单进程训练")
        use_tensor_parallel = True

    # 加载模型 - 根据选择的并行策略配置
    model_kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,  # 减少CPU内存使用
    }

    if use_tensor_parallel:
        # 张量并行配置
        model_kwargs["device_map"] = "auto"

        # 如果是多GPU环境，设置max_memory
        if num_gpus > 1:
            max_memory = {}
            for i in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                total_mem = gpu_props.total_memory
                # 预留20%内存给训练时的梯度、优化器状态等
                usable_mem = int(total_mem * 0.8)
                max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

            model_kwargs["max_memory"] = max_memory
            logger.info(f"🔧 张量并行配置:")
            logger.info(f"  device_map: auto")
            logger.info(f"  max_memory: {max_memory}")
    else:
        # DDP配置 - 不使用device_map
        logger.info("🔧 DDP配置: 不使用device_map")
        # 对于DDP，不设置device_map，让Accelerate处理设备分配

    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("✅ 模型加载完成")
    except OSError as e:
        if "tensor parallel is only supported for" in str(e):
            logger.error(f"❌ 张量并行加载失败: {e}")
            logger.info("🔄 尝试使用DDP模式重新加载...")
            # 移除张量并行相关配置
            if "device_map" in model_kwargs:
                del model_kwargs["device_map"]
            if "max_memory" in model_kwargs:
                del model_kwargs["max_memory"]

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs
            )
            logger.info("✅ 使用DDP模式加载模型成功")
        else:
            raise

    # 显示模型分布信息
    logger.info("📊 模型分布情况:")
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        logger.info("🔧 使用HuggingFace设备映射:")
        for module_name, device in model.hf_device_map.items():
            logger.info(f"  {module_name}: {device}")

        # 统计每个GPU上的模块数量
        device_count = {}
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_count[device_str] = device_count.get(device_str, 0) + 1

        logger.info("📈 设备使用统计:")
        for device, count in device_count.items():
            logger.info(f"  {device}: {count} 个模块")
    else:
        # 检查模型参数的设备分布
        device_params = {}
        total_params = 0
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_params:
                device_params[device] = {'count': 0, 'size': 0}
            device_params[device]['count'] += 1
            device_params[device]['size'] += param.numel()
            total_params += param.numel()

        logger.info("📈 参数设备分布:")
        for device, info in device_params.items():
            param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32
            percentage = info['size'] / total_params * 100
            logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

    # 显示GPU内存使用情况
    if torch.cuda.is_available():
        logger.info("💾 GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

    # 配置PEFT
    if script_args.use_peft:
        logger.info("🔧 配置LoRA")

        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            if model_args.load_in_8bit or model_args.load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)

            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit,
                                                       int8=model_args.load_in_8bit)

            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)

        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        model.print_trainable_parameters()
    else:
        logger.info("🔧 全参数训练模式")
        model = model.float()
        print_trainable_parameters(model)

    # 加载数据集
    logger.info("🔄 开始加载数据集...")
    raw_datasets = load_datasets(data_args, model_args)

    # 预处理数据集
    logger.info("🔄 开始预处理数据集...")
    preprocess_function = create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX)

    # 处理训练数据
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        tokenized_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda example: filter_empty_labels(example, IGNORE_INDEX),
            num_proc=data_args.preprocessing_num_workers
        )

        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
        replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                           for label in list(train_dataset[0]['labels'])]
        logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    # 处理验证数据
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_size = len(eval_dataset)
        logger.debug(f"Num eval_samples: {eval_size}")
        if eval_size > 500:
            logger.warning(f"Num eval_samples is large: {eval_size}, "
                           f"training slow, consider reduce it by `--max_eval_samples=50`")
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda example: filter_empty_labels(example, IGNORE_INDEX),
            num_proc=data_args.preprocessing_num_workers
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    logger.info("✅ 数据集预处理完成")

    # 设置数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    )

    # 创建数据加载器
    train_dataloader = None
    eval_dataloader = None

    if training_args.do_train and train_dataset is not None:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

    if training_args.do_eval and eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # 设置优化器
    optimizer = None
    lr_scheduler = None

    if training_args.do_train:
        # 只优化需要梯度的参数
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        # 计算总训练步数
        num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        # 设置学习率调度器
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps * training_args.warmup_ratio),
            num_training_steps=max_train_steps,
        )

    # 使用Accelerate准备所有组件 - 针对不同并行策略优化
    logger.info("🔄 开始准备训练组件...")

    # 检查模型是否已经分布在多个设备上
    model_is_distributed = hasattr(model, 'hf_device_map') and model.hf_device_map

    if model_is_distributed:
        logger.info("🔧 检测到模型已分布在多设备，使用兼容模式")
        # 对于已经分布的模型，只准备数据加载器和优化器
        if training_args.do_train:
            # 不要让accelerator包装已经分布的模型
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        # 手动设置模型的训练模式
        model.train() if training_args.do_train else model.eval()

        logger.info("✅ 分布式模型训练组件准备完成")
    else:
        logger.info("🔧 标准模式，让Accelerate处理所有组件")
        if training_args.do_train:
            model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler
            )
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            model = accelerator.prepare(model)
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        logger.info("✅ 标准训练组件准备完成")

    # 启用梯度检查点
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        # 对于DDP包装的模型，需要通过module访问原始模型的config
        if hasattr(model, "module"):
            model.module.config.use_cache = False
            logger.info("Gradient checkpointing enabled for DDP model.")
        else:
            model.config.use_cache = False
            logger.info("Gradient checkpointing enabled.")
    else:
        # 同样，对于DDP包装的模型，需要通过module访问原始模型的config
        if hasattr(model, "module"):
            model.module.config.use_cache = True
            logger.info("Gradient checkpointing disabled for DDP model.")
        else:
            model.config.use_cache = True
            logger.info("Gradient checkpointing disabled.")

    # 对于DDP包装的模型，需要通过module访问原始模型的方法
    if hasattr(model, "module"):
        model.module.enable_input_require_grads()
    else:
        model.enable_input_require_grads()

    logger.info("🎉 Accelerate多GPU训练配置成功！")

    # 开始训练
    if training_args.do_train:
        logger.info("*** 开始训练 ***")

        # 训练循环
        model.train()
        total_loss = 0
        completed_steps = 0

        # 创建进度条
        progress_bar = tqdm(
            range(int(training_args.num_train_epochs * len(train_dataloader))),
            disable=not accelerator.is_local_main_process,
            desc="Training"
        )

        for epoch in range(int(training_args.num_train_epochs)):
            logger.info(f"开始第 {epoch + 1}/{int(training_args.num_train_epochs)} 轮训练")

            for step, batch in enumerate(train_dataloader):
                # 针对张量并行优化的训练步骤
                if model_is_distributed:
                    # 分布式模型的训练步骤
                    # 前向传播
                    outputs = model(**batch)
                    loss = outputs.loss

                    # 梯度缩放（如果需要）
                    if training_args.gradient_accumulation_steps > 1:
                        loss = loss / training_args.gradient_accumulation_steps

                    # 反向传播
                    loss.backward()

                    # 梯度累积检查
                    if (step + 1) % training_args.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        if training_args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        completed_steps += 1
                        progress_bar.update(1)
                else:
                    # 标准Accelerate训练步骤
                    with accelerator.accumulate(model):
                        # 前向传播
                        outputs = model(**batch)
                        loss = outputs.loss

                        # 反向传播
                        accelerator.backward(loss)

                        # 更新参数
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        completed_steps += 1
                        progress_bar.update(1)

                # 记录损失
                total_loss += loss.detach().float()

                # 检查是否完成了一个完整的训练步骤
                step_completed = False
                if model_is_distributed:
                    step_completed = (step + 1) % training_args.gradient_accumulation_steps == 0
                else:
                    step_completed = accelerator.sync_gradients

                if step_completed:
                    # 定期记录日志
                    if completed_steps % training_args.logging_steps == 0:
                        avg_loss = total_loss / training_args.logging_steps
                        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else training_args.learning_rate
                        logger.info(f"Step {completed_steps}: loss = {avg_loss:.4f}, lr = {current_lr:.2e}")
                        total_loss = 0

                    # 定期保存检查点
                    if training_args.save_steps > 0 and completed_steps % training_args.save_steps == 0:
                        output_dir = os.path.join(training_args.output_dir, f"checkpoint-{completed_steps}")
                        if model_is_distributed:
                            # 分布式模型保存
                            os.makedirs(output_dir, exist_ok=True)
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            # 保存优化器状态
                            torch.save({
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                                'completed_steps': completed_steps,
                            }, os.path.join(output_dir, 'training_state.pt'))
                        else:
                            accelerator.save_state(output_dir)
                        logger.info(f"保存检查点到: {output_dir}")

                    # 定期评估
                    if (training_args.do_eval and
                            training_args.eval_steps > 0 and
                            completed_steps % training_args.eval_steps == 0 and
                            eval_dataloader is not None):

                        logger.info("*** 开始评估 ***")
                        model.eval()
                        eval_loss = 0
                        eval_steps = 0

                        for eval_batch in eval_dataloader:
                            with torch.no_grad():
                                eval_outputs = model(**eval_batch)
                                eval_loss += eval_outputs.loss.detach().float()
                                eval_steps += 1

                        avg_eval_loss = eval_loss / eval_steps
                        try:
                            perplexity = math.exp(avg_eval_loss)
                        except OverflowError:
                            perplexity = float("inf")

                        logger.info(
                            f"Step {completed_steps}: eval_loss = {avg_eval_loss:.4f}, perplexity = {perplexity:.2f}")
                        model.train()

        progress_bar.close()
        logger.info("✅ 训练完成")

        # 保存最终模型
        logger.info(f"保存最终模型到: {training_args.output_dir}")

        # 在训练结束后，恢复模型的use_cache设置
        unwrapped = get_unwrapped_model(model)
        unwrapped.config.use_cache = True
        unwrapped.enable_input_require_grads()

        # 保存模型时也需要考虑DDP包装
        if model_is_distributed:
            # 分布式模型直接保存
            logger.info("🔧 保存分布式模型...")
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
        else:
            # 标准Accelerate保存
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # 获取原始模型（去除包装）
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(unwrapped_model, tokenizer, training_args.output_dir)
                logger.info("✅ 标准模型保存完成")

    # 最终评估
    if training_args.do_eval and eval_dataloader is not None:
        logger.info("*** 最终评估 ***")
        model.eval()
        eval_loss = 0
        eval_steps = 0

        for eval_batch in eval_dataloader:
            with torch.no_grad():
                eval_outputs = model(**eval_batch)
                eval_loss += eval_outputs.loss.detach().float()
                eval_steps += 1

        avg_eval_loss = eval_loss / eval_steps
        try:
            perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"最终评估结果: eval_loss = {avg_eval_loss:.4f}, perplexity = {perplexity:.2f}")

    logger.info("🎉 所有任务完成！")


if __name__ == "__main__":
    main()

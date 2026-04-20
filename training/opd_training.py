# -*- coding: utf-8 -*-
"""
Standalone OPD training with TRL GKDTrainer.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Literal, Optional

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from transformers.integrations import is_deepspeed_zero3_enabled

try:
    from trl.experimental.gkd import GKDConfig, GKDTrainer
except ImportError as exc:
    raise ImportError("Standalone OPD requires trl>=0.29.0. Please upgrade TRL before running this script.") from exc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.template import Conversation, get_conv_template
from training.tool_utils import FunctionCall, get_tool_utils, load_local_json_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for student weights initialization."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for weights initialization."},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the student model in 8bit mode."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the student model in 4bit mode."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token for Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use a fast tokenizer."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": "Override the default torch.dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device map for student model loading."},
    )
    teacher_device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device map for teacher model loading."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings for the student model."},
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run OPD training.")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset name to use via the datasets library."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset configuration name to use."},
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The evaluation jsonl file folder."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples to this value if set."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "Percentage of train set used as validation set when validation is missing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for preprocessing."},
    )


@dataclass
class OPDScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    template_name: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt template name. If not set, use tokenizer's built-in chat_template."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for agent training. Options: default, glm4, llama3, mistral, qwen."},
    )
    max_prompt_length: int = field(default=1024, metadata={"help": "Maximum prompt token length."})
    opd_lambda: float = field(default=0.5, metadata={"help": "On-policy rollout ratio."})
    opd_beta: float = field(default=0.5, metadata={"help": "JSD / KL interpolation coefficient."})
    teacher_load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load teacher model in 8bit."})
    teacher_load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load teacher model in 4bit."})

    def __post_init__(self):
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive.")
        if not 0.0 <= self.opd_lambda <= 1.0:
            raise ValueError("opd_lambda must be in the range [0.0, 1.0].")
        if not 0.0 <= self.opd_beta <= 1.0:
            raise ValueError("opd_beta must be in the range [0.0, 1.0].")


def format_function_call_value(value: str, tool_format: Optional[str]) -> str:
    fc_dict = json.loads(value)
    if "name" not in fc_dict or "arguments" not in fc_dict:
        return value
    if tool_format:
        tool_utils = get_tool_utils(tool_format)
        return tool_utils.function_formatter(
            [FunctionCall(fc_dict["name"], json.dumps(fc_dict["arguments"], ensure_ascii=False))]
        )
    return f"Action: {fc_dict['name']}\nAction Input: {json.dumps(fc_dict['arguments'], ensure_ascii=False)}"


def format_observation_value(value: str, tool_format: Optional[str]) -> str:
    if tool_format == "qwen":
        return f"<tool_response>\n{value}\n</tool_response>"
    if tool_format == "glm4":
        return f"<|observation|>\n{value}"
    if tool_format == "mistral":
        return f'[TOOL_RESULTS] {{"content": {value}}}[/TOOL_RESULTS]'
    return f"Observation: {value}"


def build_chat_messages(
    source: List[Dict[str, Any]],
    tools_json: Optional[str] = None,
    tool_format: Optional[str] = None,
    system_prompt_override: str = "",
) -> List[Dict[str, str]]:
    system_prompt = system_prompt_override
    tools_text = ""

    if tools_json:
        tools_parsed = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
        if tools_parsed and tool_format:
            tool_utils = get_tool_utils(tool_format)
            tools_text = tool_utils.tool_formatter(tools_parsed)

    messages: List[Dict[str, str]] = []
    for sentence in source:
        role = sentence.get("from", "")
        value = sentence.get("value", "")

        if role == "system":
            system_prompt = value
            continue

        if role in ["human", "user", "observation"]:
            if role == "observation":
                value = format_observation_value(value, tool_format)
            messages.append({"role": "user", "content": value})
            continue

        if role in ["gpt", "assistant", "function_call"]:
            if role == "function_call":
                value = format_function_call_value(value, tool_format)
            messages.append({"role": "assistant", "content": value})

    if tools_text:
        system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text
    if system_prompt:
        return [{"role": "system", "content": system_prompt}] + messages
    return messages


def convert_sharegpt_batch_to_messages(
    examples: Dict[str, List[Any]],
    tool_format: Optional[str] = None,
) -> Dict[str, List[List[Dict[str, str]]]]:
    batch_messages: List[List[Dict[str, str]]] = []
    tools_list = examples.get("tools")

    for index, source in enumerate(examples["conversations"]):
        tools_json = tools_list[index] if tools_list is not None else None
        chat_messages = build_chat_messages(source, tools_json=tools_json, tool_format=tool_format)
        if not chat_messages:
            continue

        prefix: List[Dict[str, str]] = []
        saw_user = False
        for message in chat_messages:
            prefix.append(message)
            if message["role"] == "user":
                saw_user = True
                continue
            if message["role"] == "assistant" and saw_user:
                batch_messages.append([item.copy() for item in prefix])

    return {"messages": batch_messages}


def resolve_torch_dtype(torch_dtype: Optional[str]):
    if torch_dtype in ["auto", None]:
        return torch_dtype
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def find_all_linear_names(peft_model, int4: bool = False, int8: bool = False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes  # type: ignore[import-not-found]

        if int4:
            cls = bitsandbytes.nn.Linear4bit
        else:
            cls = bitsandbytes.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if "lm_head" in name or "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool, qlora: bool, torch_dtype):
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot be set at the same time.")
    if not load_in_4bit and not load_in_8bit:
        return None
    if is_deepspeed_zero3_enabled():
        raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    if qlora:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )


def load_raw_datasets(data_args: DataArguments, cache_dir: Optional[str]):
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=cache_dir,
        )
        if "validation" not in raw_datasets:
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42,
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
        return raw_datasets

    data_files = {}
    if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
        train_data_files = glob(f"{data_args.train_file_dir}/**/*.jsonl", recursive=True)
        logger.info(f"train files: {train_data_files}")
        data_files["train"] = train_data_files
    if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
        eval_data_files = glob(f"{data_args.validation_file_dir}/**/*.jsonl", recursive=True)
        logger.info(f"eval files: {eval_data_files}")
        data_files["validation"] = eval_data_files

    raw_datasets = load_local_json_datasets(data_files, cache_dir=cache_dir)
    if "validation" not in raw_datasets:
        shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
        split = shuffled_train_dataset.train_test_split(
            test_size=float(data_args.validation_split_percentage / 100),
            seed=42,
        )
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    return raw_datasets


def normalize_messages_batch(examples, tool_format: Optional[str]):
    if "messages" in examples:
        normalized_messages = []
        for message_list in examples["messages"]:
            if not message_list:
                continue
            if message_list[-1]["role"] != "assistant":
                continue
            normalized_messages.append(
                [{"role": message["role"], "content": message["content"]} for message in message_list]
            )
        return {"messages": normalized_messages}

    if "conversations" not in examples:
        raise ValueError("OPD expects a dataset with either `messages` or `conversations` columns.")
    return convert_sharegpt_batch_to_messages(examples, tool_format=tool_format)


def build_history_pairs(messages):
    system_prompt = ""
    start_index = 0
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        start_index = 1

    history_pairs = []
    current_user = None
    for message in messages[start_index:]:
        if message["role"] == "user":
            if current_user is None:
                current_user = message["content"]
            else:
                current_user += "\n" + message["content"]
        elif message["role"] == "assistant" and current_user is not None:
            history_pairs.append([current_user, message["content"]])
            current_user = None
    return system_prompt, history_pairs


def build_template_text(messages, tokenizer, prompt_template: Conversation):
    system_prompt, history_pairs = build_history_pairs(messages)
    if not history_pairs:
        return None

    dialog = prompt_template.get_dialog(history_pairs, system_prompt=system_prompt)
    prompt_text = "".join(dialog[:-1])
    full_text = prompt_text + dialog[-1]

    if tokenizer.bos_token is not None and not full_text.startswith(tokenizer.bos_token):
        prompt_text = tokenizer.bos_token + prompt_text
        full_text = tokenizer.bos_token + full_text
    if tokenizer.eos_token is not None and not full_text.endswith(tokenizer.eos_token):
        full_text = full_text + tokenizer.eos_token

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    return {
        "prompt": prompt_text,
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "prompt_length": len(tokenized_prompt["input_ids"]),
    }


def build_chatml_prompt(messages, tokenizer):
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    return {
        "prompt": prompt_text,
        "prompt_length": len(tokenized_prompt["input_ids"]),
    }


def prepare_dataset_split(
    dataset: Dataset,
    tokenizer,
    script_args: OPDScriptArguments,
    prompt_template: Optional[Conversation],
    do_truncate: bool,
    num_proc: Optional[int],
    overwrite_cache: bool,
):
    needs_manual_template = prompt_template is not None
    if prompt_template is None and tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not provide chat_template. Please pass --template_name for OPD training.")

    def preprocess_function(examples):
        normalized = normalize_messages_batch(examples, script_args.tool_format)
        outputs = {"messages": [], "prompt": []}
        if needs_manual_template:
            outputs["input_ids"] = []
            outputs["attention_mask"] = []

        for messages in normalized["messages"]:
            if len(messages) < 2:
                continue
            if needs_manual_template:
                rendered = build_template_text(messages, tokenizer, prompt_template)
            else:
                rendered = build_chatml_prompt(messages, tokenizer)
            if rendered is None:
                continue
            if rendered["prompt_length"] > script_args.max_prompt_length:
                continue

            outputs["messages"].append(messages)
            outputs["prompt"].append(rendered["prompt"])
            if needs_manual_template:
                outputs["input_ids"].append(rendered["input_ids"])
                outputs["attention_mask"].append(rendered["attention_mask"])
        return outputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Preparing OPD dataset" if do_truncate else None,
    )

    return processed_dataset.filter(lambda example: len(example["prompt"]) > 0, num_proc=num_proc)


def load_student_model(model_args: ModelArguments, training_args: GKDConfig, script_args: OPDScriptArguments, torch_dtype):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    device_map = None if ddp or model_args.device_map in ["None", "none", ""] else model_args.device_map

    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    if model_args.rope_scaling is not None:
        current_max_length = config.max_position_embeddings
        if script_args.max_prompt_length > current_max_length:
            scaling_factor = float((script_args.max_prompt_length + current_max_length - 1) // current_max_length)
        else:
            scaling_factor = 1.0
        config.rope_scaling = {"type": model_args.rope_scaling, "factor": scaling_factor}
    quantization_config = build_quantization_config(
        model_args.load_in_4bit,
        model_args.load_in_8bit,
        script_args.qlora,
        torch_dtype,
    )

    model_kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    if model_args.flash_attn:
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if script_args.use_peft:
        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            if quantization_config is not None:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(",") if script_args.target_modules else None
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(
                    model,
                    int4=model_args.load_in_4bit,
                    int8=model_args.load_in_8bit,
                )
            modules_to_save = script_args.modules_to_save.split(",") if script_args.modules_to_save else None
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = model.float()

    if training_args.gradient_checkpointing and model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def load_teacher_model(
    model_args: ModelArguments,
    training_args: GKDConfig,
    script_args: OPDScriptArguments,
    torch_dtype,
):
    teacher_model_name_or_path = training_args.teacher_model_name_or_path or model_args.model_name_or_path
    teacher_device_map = None if model_args.teacher_device_map in ["None", "none", ""] else model_args.teacher_device_map
    teacher_quantization_config = build_quantization_config(
        script_args.teacher_load_in_4bit,
        script_args.teacher_load_in_8bit,
        False,
        torch_dtype,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=teacher_quantization_config,
        low_cpu_mem_usage=True,
        device_map=teacher_device_map,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.hf_hub_token,
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False
    return teacher_model


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GKDConfig, OPDScriptArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, script_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(
            look_for_args_file=False
        )

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"GKD args: {training_args}")
    logger.info(f"Script args: {script_args}")

    set_seed(training_args.seed)
    torch_dtype = resolve_torch_dtype(model_args.torch_dtype)

    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        token=model_args.hf_hub_token,
        padding_side="left",
    )
    prompt_template = get_conv_template(script_args.template_name) if script_args.template_name else None
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str if prompt_template else "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token_id is not None else tokenizer.eos_token

    raw_datasets = load_raw_datasets(data_args, model_args.cache_dir)
    logger.info(f"Raw datasets: {raw_datasets}")

    train_dataset = None
    eval_dataset = None
    max_train_samples = 0
    max_eval_samples = 0

    if training_args.do_train:
        train_dataset = raw_datasets["train"].shuffle(seed=42)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        max_train_samples = len(train_dataset)
        train_dataset = prepare_dataset_split(
            dataset=train_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            prompt_template=prompt_template,
            do_truncate=True,
            num_proc=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
        )
        logger.info(f"Prepared train dataset size: {len(train_dataset)}")

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        max_eval_samples = len(eval_dataset)
        eval_dataset = prepare_dataset_split(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            prompt_template=prompt_template,
            do_truncate=False,
            num_proc=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
        )
        logger.info(f"Prepared eval dataset size: {len(eval_dataset)}")

    training_args.max_length = script_args.max_prompt_length + training_args.max_new_tokens
    training_args.lmbda = script_args.opd_lambda
    training_args.beta = script_args.opd_beta
    if training_args.teacher_model_name_or_path is None:
        training_args.teacher_model_name_or_path = model_args.model_name_or_path
    training_args.remove_unused_columns = False

    student_model = load_student_model(model_args, training_args, script_args, torch_dtype)
    teacher_model = load_teacher_model(model_args, training_args, script_args, torch_dtype)

    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

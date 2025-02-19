# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using DPO
"""
import os
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import DPOTrainer

from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with DPO
    """
    # Model arguments
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "Train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Eval batch size per device"})
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # Training arguments
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "The number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "The weight decay"})
    optim: Optional[str] = field(default="adamw_hf", metadata={"help": "The optimizer type"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Whether to use fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "The number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "X steps to evaluate the model"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "X steps to log the model"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "The output directory"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


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
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Parse args: {args}")

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = get_conv_template(args.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # Get datasets
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        """Load the paired dataset and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        Prompts are structured as follows:
          system_prompt + history[[q,a], [q,a]...] + question
        """
        prompts = []
        for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
            system_prompt = system or ""
            history_with_question = history + [[question, '']] if history else [[question, '']]
            prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("First train example:")
        first_example = train_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("First eval example:")
        first_example = eval_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    # Load model
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    logger.info(f"Device map: {args.device_map}")
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(f"Quantizing model, load_in_4bit: {args.load_in_4bit}, load_in_8bit: {args.load_in_8bit}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )
    # fixed FP16 ValueError
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # Initialize our Trainer
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=args.remove_unused_columns,
        run_name=f"dpo_v1",
    )

    # Initialize DPO trainer
    peft_config = None
    if args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")
    trainer = DPOTrainer(
        model,
        ref_model=None if args.use_peft else deepcopy(model),
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config if args.use_peft else None,
        max_prompt_length=args.max_source_length,
        max_length=full_max_length,
    )
    print_trainable_parameters(trainer.model)

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            trainer.model.save_pretrained(args.output_dir)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using PPO
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForSequenceClassification,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import send_example_telemetry
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The reward model name"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
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

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")
        if self.reward_model_name_or_path is None:
            raise ValueError("You must specify a valid reward_model_name_or_path to run training.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
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
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class PeftArguments(TrainingArguments):
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)

    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "Whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0, metadata={"help": "Baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


DEFAULT_PAD_TOKEN = "[PAD]"
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)


def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def find_all_linear_names(peft_model, int4=False, int8=False):
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
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def get_reward_score(reward_model, reward_tokenizer, question, answer):
    inputs = reward_tokenizer(question, answer, return_tensors='pt')
    score = reward_model(**inputs).logits[0].cpu().detach()

    return score


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_rl", model_args, data_args)

    logger.warning(f"Model args: {model_args}")
    logger.warning(f"Data args: {data_args}")
    logger.warning(f"Training args: {training_args}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    # Required for llama
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    logger.info("Init new peft model")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=training_args.target_modules,
        inference_mode=False,
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
    )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        trust_remote_code=model_args.trust_remote_code,
        peft_config=peft_config,
    )
    # Load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.reward_model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(model_args.reward_model_name_or_path, **tokenizer_kwargs)

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for instruction, input in zip(examples['instruction'], examples['input']):
            if input:
                instruction = instruction + "\n" + input
            source = PROMPT_TEMPLATE.format_map({"instruction": instruction})
            tokenized_question = tokenizer(source, truncation=True, max_length=max_source_length)
            new_examples["query"].append(source)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized_dataset = train_dataset.shuffle().map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            train_dataset = tokenized_dataset.filter(
                lambda x: len(x['input_ids']) > 0
            )
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            tokenized_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = tokenized_dataset.filter(
                lambda x: len(x['input_ids']) > 0
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    output_dir = training_args.output_dir
    config = PPOConfig(
        steps=training_args.max_steps,
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        log_with=training_args.report_to,
        batch_size=training_args.per_device_train_batch_size,
        mini_batch_size=training_args.mini_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=training_args.early_stopping,
        target_kl=training_args.target_kl,
        seed=training_args.seed,
        init_kl_coef=training_args.init_kl_coef,
        adap_kl_ctrl=training_args.adap_kl_ctrl,
        accelerator_kwargs={"project_dir": output_dir},
    )
    # Set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    # These arguments are passed to the `generate` function of the PPOTrainer
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_length_sampler = LengthSampler(data_args.min_target_length, max_target_length)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        for step, batch in tqdm(enumerate(trainer.dataloader)):
            if step >= config.total_ppo_epochs:
                break
            question_tensors = batch["input_ids"]

            response_tensors = trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute reward score
            score_outputs = [
                get_reward_score(reward_model, reward_tokenizer, q, r) for q, r in
                zip(batch["query"], batch["response"])
            ]
            rewards = [torch.tensor(float(score) - training_args.reward_baseline) for score in score_outputs]

            # Run PPO step
            stats = trainer.step(question_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)

            if step and step % training_args.save_steps == 0:
                trainer.save_pretrained(os.path.join(output_dir, f"checkpoint-{step}"))
        # Save model and tokenizer
        trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

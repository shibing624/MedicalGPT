# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

usage:
accelerate launch --multi_gpu --num_machines 1  --num_processes 8 rlhf/rl_training.py --log_with=wandb
--model_name=<LLAMA_SFT_MODEL> --reward_model_name=<LLAMA_SFT_RM_MODEL>
--tokenizer_name=<LLAMA_TOKENIZER> --save_freq=100 --output_max_length=128 --batch_size=8
--gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5
--early_stopping=True --output_dir=llama-sft-rl-finetune-128
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import datasets
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from peft import TaskType, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from transformers import (
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    reward_model_name: Optional[str] = field(default="", metadata={"help": "The reward model name"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
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
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
    output_dir: str = field(
        default="./outputs-rl",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "Max length of input text"})
    output_max_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
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
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for evaluation."})
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1.41e-5)
    weight_decay: Optional[int] = field(default=0.001)
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "Run an evaluation every X steps."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy to use."})
    save_steps: Optional[int] = field(default=500, metadata={"help": "Run an save every X steps."})
    save_strategy: str = field(default="steps", metadata={"help": "The save strategy to use."})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limit the total amount of checkpoints. "})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    optim: Optional[str] = field(default="adamw_hf", metadata={"help": "The optimizer to use."})
    report_to: Optional[str] = field(
        default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "Whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "Baseline value that is subtracted from the reward"},
    )


@dataclass
class PeftArguments(TrainingArguments):
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)


logger = logging.getLogger(__name__)
DEFAULT_PAD_TOKEN = "[PAD]"


class RLTrainer(Trainer):
    """
    Trainer for RL models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def resize_model_embeddings(model, tokenizer_vocab_size):
    """Resizes model embeddings to match the tokenizer vocab size."""
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    if model_vocab_size != tokenizer_vocab_size:
        logger.info(
            f"Resize model embeddings to fit tokenizer, "
            f"Vocab of the base model: {model_vocab_size}, "
            f"Vocab of the tokenizer: {tokenizer_vocab_size}"
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
        logger.info(f"Model token embeddings updated, size: {tokenizer_vocab_size}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_rl", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.warning(f"Training args: {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Required for llama
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            torch_dtype=torch_dtype,
            device_map=model_args.device_map,
        )
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")
    # Resizes model embeddings to match the tokenizer vocab size
    resize_model_embeddings(model, tokenizer)
    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        logger.info("Init new peft model")
        target_modules = training_args.target_modules.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        logger.info(f"Peft target_modules: {target_modules}")
        logger.info(f"Peft lora_rank: {training_args.lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
    # Tokenize the dataset
    max_length = data_args.max_length  # Set max length of input text

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    # Preprocess the dataset
    train_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
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
                lambda x: 0 < len(x['input_ids']) <= max_length
            )
            logger.debug(f"Num train_samples  {len(train_dataset)}")
            logger.debug("Training example:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    eval_dataset = None
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            tokenized_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = tokenized_dataset.filter(
                lambda x: 0 < len(x['input_ids']) <= max_length
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        log_with=training_args.report_to,
        batch_size=training_args.per_device_train_batch_size,
        mini_batch_size=training_args.mini_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=training_args.early_stopping,
        target_kl=training_args.target_kl,
        ppo_epochs=training_args.num_train_epoch,
        seed=training_args.seed,
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

    current_device = Accelerator().local_process_index
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_args.reward_model_name,
        device_map={"": current_device},
        model_kwargs={"load_in_8bit": model_args.load_in_8bit},
        tokenizer=tokenizer,
    )

    # These arguments are passed to the `generate` function of the PPOTrainer
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 4
    output_length_sampler = LengthSampler(output_min_length, training_args.output_max_length)
    # We then define the arguments to pass to the sentiment analysis pipeline.
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none",
                   "batch_size": training_args.per_device_train_batch_size, "truncation": True}
    output_dir = training_args.output_dir

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        for step, batch in tqdm(enumerate(trainer.dataloader)):
            question_tensors = batch["input_ids"]

            response_tensors = trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[0]["score"] - training_args.reward_baseline) for output in pipe_outputs]

            # Run PPO step
            stats = trainer.step(question_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)

            if step and step % training_args.save_steps == 0:
                trainer.save_pretrained(os.path.join(output_dir, f"checkpoint-{step}"))
        # Save model and tokenizer
        trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

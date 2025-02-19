# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using PPO
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_peft_config,
)
from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class PPOArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})


def main():
    parser = HfArgumentParser((PPOArguments, ScriptArguments, PPOConfig, ModelConfig))
    args, script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Add distributed training initialization
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    # Only log on main process
    if is_main_process:
        logger.info(f"Parse args: {args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Model args: {model_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
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

    # Load model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # Get datasets
    prompt_template = get_conv_template(args.template_name)
    if script_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            script_args.dataset_name,
            script_args.dataset_config,
            split=script_args.dataset_train_split
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
        dataset = load_dataset(
            'json',
            data_files=data_files,
        )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                if len(source) < 2:
                    continue
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
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                system_prompt = system_prompts[i] if system_prompts else None
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)

        for dialog in get_dialog(examples):
            for i in range(len(dialog) // 2):
                source_txt = dialog[2 * i]
                tokenized_question = tokenizer(
                    source_txt, truncation=True, max_length=max_source_length, padding="max_length",
                    return_tensors="pt"
                )
                new_examples["query"].append(source_txt)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    # Preprocess the dataset
    if is_main_process:
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )
        train_dataset = tokenized_train_dataset.filter(
            lambda x: len(x['input_ids']) > 0
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        tokenized_eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )
        eval_dataset = tokenized_eval_dataset.filter(
            lambda x: len(x['input_ids']) > 0
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Training
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
        trainer.train()

        # Only log on main process
        if is_main_process:
            trainer.save_model(training_args.output_dir)

    trainer.generate_completions()


if __name__ == "__main__":
    main()

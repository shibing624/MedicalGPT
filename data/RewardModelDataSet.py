from algorithm.llm.data.BaseDataSet import BaseDataSet


class RewardModelDataSet(BaseDataSet):
    def __init__(self, **kwargs):
        super(RewardModelDataSet, self).__init__(**kwargs)

    def prepare_datasets(self, raw_datasets):
        full_max_length = self.data_args.max_source_length + self.data_args.max_target_length
        train_dataset = None
        max_train_samples = 0
        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets['train']
            max_train_samples = len(train_dataset)
            if self.data_args.max_train_samples is not None and self.data_args.max_train_samples > 0:
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            self.logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
            with self.training_args.main_process_first(desc="Train dataset tokenization"):
                tokenized_dataset = train_dataset.shuffle().map(
                    self.preprocess_reward_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                train_dataset = tokenized_dataset.filter(
                    lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                        x['input_ids_chosen']) <= full_max_length
                )
                self.logger.debug(f"Num train_samples: {len(train_dataset)}")
                self.logger.debug("Tokenized training example:")
                self.logger.debug(self.tokenizer.decode(train_dataset[0]['input_ids_chosen']))

        eval_dataset = None
        max_eval_samples = 0
        if self.training_args.do_eval:
            with self.training_args.main_process_first(desc="Eval dataset tokenization"):
                if "validation" not in raw_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = raw_datasets["validation"]
                max_eval_samples = len(eval_dataset)
                if self.data_args.max_eval_samples is not None and self.data_args.max_eval_samples > 0:
                    max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                self.logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
                tokenized_dataset = eval_dataset.map(
                    self.preprocess_reward_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                eval_dataset = tokenized_dataset.filter(
                    lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                        x['input_ids_chosen']) <= full_max_length
                )
                self.logger.debug(f"Num eval_samples: {len(eval_dataset)}")
                self.logger.debug("Tokenized eval example:")
                self.logger.debug(self.tokenizer.decode(eval_dataset[0]['input_ids_chosen']))
        return train_dataset, max_train_samples, eval_dataset, max_eval_samples

    def preprocess_reward_function(self, examples):
        """
            Turn the dataset into pairs of Question + Answer, where input_ids_chosen is the preferred question + answer
                and text_rejected is the other.
            """
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for question, chosen, rejected in zip(examples["question"], examples["response_chosen"],
                                              examples["response_rejected"]):
            tokenized_chosen = self.tokenizer("Question: " + question + "\n\nAnswer: " + chosen)
            tokenized_rejected = self.tokenizer("Question: " + question + "\n\nAnswer: " + rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

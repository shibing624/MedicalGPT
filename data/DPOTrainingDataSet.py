from algorithm.llm.data.BaseDataSet import BaseDataSet


class DPOTrainingDataSet(BaseDataSet):
    def __init__(self, **kwargs):
        super(DPOTrainingDataSet, self).__init__(**kwargs)

    def prepare_datasets(self, raw_datasets):
        max_source_length = self.data_args.max_source_length
        max_target_length = self.data_args.max_target_length
        full_max_length = max_source_length + max_target_length

        # Preprocess the dataset
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
            tokenized_dataset = train_dataset.shuffle().map(
                self.return_prompt_and_responses,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            train_dataset = tokenized_dataset.filter(
                lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                          and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
            )
            self.logger.debug(f"Num train_samples: {len(train_dataset)}")
            self.logger.debug("First train example:")
            self.logger.debug(train_dataset[0]['prompt'] + train_dataset[0]['chosen'])

        eval_dataset = None
        max_eval_samples = 0
        if self.training_args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if self.data_args.max_eval_samples is not None and self.data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            self.logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                self.return_prompt_and_responses,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = eval_dataset.filter(
                lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                          and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
            )
            self.logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            self.logger.debug("First eval example:")
            self.logger.debug(eval_dataset[0]['prompt'] + eval_dataset[0]['chosen'])
            return train_dataset, max_train_samples, eval_dataset, max_eval_samples

    def return_prompt_and_responses(self, examples):

        """Load the paired dataset and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        Prompts are structured as follows:
          "Question: " + <prompt> + "\n\nAnswer: "
        """
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in examples["question"]],
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

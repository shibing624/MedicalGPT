from itertools import chain

from algorithm.llm.data.BaseDataSet import BaseDataSet


class PreTrainingDataSet(BaseDataSet):
    def __init__(self, **kwargs):
        super(PreTrainingDataSet, self).__init__(**kwargs)
        # self.data_args = kwargs["data_args"]
        # self.model_args = kwargs["model_args"]
        # self.logger = kwargs["logger"]
        # self.training_args = kwargs["training_args"]
        # self.tokenizer = kwargs["tokenizer"]

    def prepare_datasets(self, raw_datasets, block_size):
        # Preprocessing the datasets.

        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        if self.training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        with self.training_args.main_process_first(desc="Dataset tokenization and grouping"):
            if not self.data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
        train_dataset = None
        max_train_samples = 0
        if self.training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = lm_datasets['train']
            max_train_samples = len(train_dataset)
            if self.data_args.max_train_samples is not None and self.data_args.max_train_samples > 0:
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            self.logger.debug(f"Num train_samples: {len(train_dataset)}")
            self.logger.debug("Tokenized training example:")
            self.logger.debug(self.tokenizer.decode(train_dataset[0]['input_ids']))

        eval_dataset = None
        max_eval_samples = 0
        if self.training_args.do_eval:
            if "validation" not in tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = lm_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if self.data_args.max_eval_samples is not None and self.data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            self.logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            self.logger.debug("Tokenized eval example:")
            self.logger.debug(self.tokenizer.decode(eval_dataset[0]['input_ids']))
        return train_dataset, max_train_samples, eval_dataset, max_eval_samples

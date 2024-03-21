from algorithm.llm.data.BaseDataSet import BaseDataSet
from algorithm.llm.data.template.DataTemplate import get_conv_template
class PPOTrainingDataSet(BaseDataSet):

    def __init__(self, **kwargs):
        super(PPOTrainingDataSet, self).__init__(**kwargs)


    def prepare_datasets(self, raw_datasets):
        train_dataset = None
        max_train_samples = 0
        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets['train']
            if self.data_args.max_train_samples is not None and self.data_args.max_train_samples > 0:
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            self.logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
            tokenized_dataset = train_dataset.shuffle().map(
                self.preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            train_dataset = tokenized_dataset.filter(
                lambda x: len(x['input_ids']) > 0
            )
            self.logger.debug(f"Num train_samples: {len(train_dataset)}")
        return train_dataset, max_train_samples, None, None

    def preprocess_function(self, examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        roles = ["human", "gpt"]

        def get_prompt(examples):
            prompt_template = get_conv_template(self.data_args.template_name)
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
                        self.logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                yield prompt_template.get_prompt(history_messages)

        for prompt in get_prompt(examples):
            for i in range(len(prompt) // 2):
                source_txt = prompt[2 * i]
                tokenized_question = self.tokenizer(
                    source_txt, truncation=True, max_length=self.data_args.max_source_length, padding="max_length",
                    return_tensors="pt"
                )
                new_examples["query"].append(source_txt)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples
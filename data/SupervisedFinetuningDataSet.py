from algorithm.llm.data.BaseDataSet import BaseDataSet
from algorithm.llm.data.template.DataTemplate import get_conv_template
from transformers.trainer_pt_utils import LabelSmoother


class SupervisedFinetuningDataSet(BaseDataSet):
    def __init__(self, **kwargs):
        super(SupervisedFinetuningDataSet, self).__init__(**kwargs)
        self.IGNORE_INDEX = LabelSmoother.ignore_index if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
    def prepare_datasets(self, raw_datasets):
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
                train_dataset = train_dataset.shuffle().map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                train_dataset = train_dataset.filter(self.filter_empty_labels,
                                                     num_proc=self.data_args.preprocessing_num_workers)
                self.logger.debug(f"Num train_samples: {len(train_dataset)}")
                self.logger.debug("Tokenized training example:")
                self.logger.debug(f"Decode input_ids[0]: {self.tokenizer.decode(train_dataset[0]['input_ids'])}")
                replaced_labels = [label if label != self.IGNORE_INDEX else self.tokenizer.pad_token_id
                                   for label in list(train_dataset[0]['labels'])]
                self.logger.debug(f"Decode labels[0]: {self.tokenizer.decode(replaced_labels)}")

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
                eval_dataset = eval_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
                eval_dataset = eval_dataset.filter(self.filter_empty_labels,
                                                   num_proc=self.data_args.preprocessing_num_workers)
                self.logger.debug(f"Num eval_samples: {len(eval_dataset)}")
                self.logger.debug("Tokenized eval example:")
                self.logger.debug(self.tokenizer.decode(eval_dataset[0]['input_ids']))
        return train_dataset, max_train_samples, eval_dataset, max_eval_samples

    def preprocess_function(self, examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        prompt_template = get_conv_template(self.data_args.template_name)
        max_length = self.script_args.model_max_length
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
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
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                yield prompt_template.get_dialog(history_messages)

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = self.tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = self.tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == self.tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [self.tokenizer.eos_token_id]  # add eos token for each turn
                labels += [self.IGNORE_INDEX] * len(source_ids) + target_ids + [self.tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(self, example):
        """Remove empty labels dataset."""
        return not all(label == self.IGNORE_INDEX for label in example["labels"])

    def get_ignore_index(self):
        return self.IGNORE_INDEX
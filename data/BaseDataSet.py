from datasets import load_dataset
import os
from glob import glob


class BaseDataSet(object):

    def __init__(self, **kwargs):
        self.data_args = kwargs["data_args"]
        self.model_args = kwargs["model_args"]
        self.logger = kwargs["logger"]
        self.training_args = kwargs["training_args"]
        self.tokenizer = kwargs["tokenizer"]
        self.script_args = kwargs["script_args"]


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    def load_raw_data(self, file_suffix=["*.txt", "*.json", "*.jsonl"]):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
                streaming=self.data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    streaming=self.data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    streaming=self.data_args.streaming,
                )
        else:
            data_files = {}
            dataset_args = {}
            if self.data_args.train_file_dir is not None and os.path.exists(self.data_args.train_file_dir):
                train_data_files = []
                for file in file_suffix:
                    train_data_files += glob(f'{self.data_args.train_file_dir}/**/' + file, recursive=True)
                self.logger.info(f"train files: {train_data_files}")
                # Train data files must be same type, e.g. all txt or all jsonl
                types = [f.split('.')[-1] for f in train_data_files]
                if len(set(types)) > 1:
                    raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
                data_files["train"] = train_data_files
            if self.data_args.validation_file_dir is not None and os.path.exists(self.data_args.validation_file_dir):
                eval_data_files = []
                for file in file_suffix:
                    eval_data_files += glob(f'{self.data_args.train_file_dir}/**/' + file, recursive=True)
                self.logger.info(f"eval files: {eval_data_files}")
                data_files["validation"] = eval_data_files
                # Train data files must be same type, e.g. all txt or all jsonl
                types = [f.split('.')[-1] for f in eval_data_files]
                if len(set(types)) > 1:
                    raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
            extension = "text" if data_files["train"][0].endswith('txt') else 'json'
            if extension == "text":
                dataset_args["keep_linebreaks"] = self.data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                **dataset_args,
            )

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    **dataset_args,
                )
        self.logger.info(f"Raw datasets: {raw_datasets}")
        return raw_datasets

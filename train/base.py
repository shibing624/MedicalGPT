from loguru import logger
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from algorithm.llm.arguments.ModelArguments import ModelArguments
from algorithm.llm.arguments.DataArguments import DataArguments
from algorithm.llm.arguments.ScriptArguments import ScriptArguments
from algorithm.llm.model.Model import MODEL_CLASSES
from algorithm.llm.utils.EnvUtils import use_modelscope
import os


class Base(object):

    def __init__(self):
        pass

    def initial_train(self, TraingingArguments=Seq2SeqTrainingArguments):
        parser = HfArgumentParser((ModelArguments, DataArguments, TraingingArguments, ScriptArguments))
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

        return model_args, data_args, training_args, script_args

    def load_tokenizer(self, model_args):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "trust_remote_code": model_args.trust_remote_code,
        }
        self.try_download_model_from_ms(model_args)
        tokenizer_name_or_path = model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path \
            else model_args.model_name_or_path

        tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        return tokenizer

    def try_download_model_from_ms(self, model_args):
        if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
            return
        try:
            from modelscope import snapshot_download
            # revision = "master" if model_args.model_revision == "main" else model_args.model_revision
            model_args.model_name_or_path = snapshot_download(
                model_args.model_name_or_path, revision="master", cache_dir=model_args.cache_dir
            )
            if model_args.tokenizer_name_or_path:
                model_args.tokenizer_name_or_path = snapshot_download(
                model_args.tokenizer_name_or_path, revision="master", cache_dir=model_args.cache_dir
            )
        except ImportError:
            raise ImportError("Please install modelscope via `pip install modelscope -U`")

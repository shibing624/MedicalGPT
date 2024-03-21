import os
from loguru import logger
from algorithm.llm.data.PreTrainingDataSet import PreTrainingDataSet
from algorithm.llm.model.Model import Model
from algorithm.llm.train.base import Base
from settings import ConfigPackage
from settings import SettingPackage


class PreTraining(Base):
    name = "pretraining"
    config_file = "%s%s" % (name, "_config")
    setting_file = "%s%s" % (name, "_setting")

    def __init__(self, config_file_input=None, setting_file_input=None, log=None):
        self.config_file = config_file_input if config_file_input else \
            "%s%s" % (ConfigPackage, self.config_file)

        self.setting_file = setting_file_input if setting_file_input else \
            "%s%s" % (SettingPackage, self.setting_file)
        if os.path.exists(self.config_file):
            self.config_package = __import__(self.config_file, fromlist=True)
            self.basic_config = getattr(self.config_package, "basic_config")
            if not os.path.exists(self.basic_config["out"]):
                os.makedirs(self.basic_config["out"])

        self.log = log

    def before_run(self):
        print("开始运行:", PreTraining.name)

        # 初始化训练器

    def run(self):
        model_args, data_args, training_args, script_args = self.initial_train()
        # 加载tokenizer
        tokenizer = self.load_tokenizer(model_args=model_args)
        # 加载数据集
        dataset_class = PreTrainingDataSet(data_args=data_args, script_args=script_args, model_args=model_args,
                                           logger=logger, training_args=training_args,
                                           tokenizer=tokenizer)
        raw_datasets = dataset_class.load_raw_data()
        block_size = self.define_block_size(data_args=data_args, tokenizer=tokenizer)
        train_dataset, max_train_samples, eval_dataset, max_eval_samples = dataset_class.prepare_datasets(
            raw_datasets, block_size=block_size)

        # 加载模型
        model_class = Model(model_args=model_args, script_args=script_args, data_args=data_args,
                            training_args=training_args, logger=logger)
        model = model_class.load_model()
        if script_args.use_peft:
            model = model_class.fine_tuning_model(model, tokenizer)
        else:
            model = model_class.full_training_model(model)
        if training_args.do_train:
            model_class.train(model, tokenizer, train_dataset, eval_dataset, max_train_samples)
        if training_args.do_eval:
            model_class.evaluate(model, tokenizer, train_dataset, eval_dataset, max_eval_samples)

    def define_block_size(self, data_args, tokenizer):
        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 2048:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)
        return block_size

    def after_run(self):
        print("结束运行:", PreTraining.name)
        pass


if __name__ == '__main__':
    pre_training = PreTraining()
    pre_training.run()

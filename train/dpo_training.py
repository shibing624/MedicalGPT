from algorithm.llm.train.base import Base
from settings import ConfigPackage
from settings import SettingPackage
import os
from loguru import logger
from algorithm.llm.data.DPOTrainingDataSet import DPOTrainingDataSet
from algorithm.llm.model.DPOTrainingModel import DPOTrainingModel
from algorithm.llm.trainer.DPOTrainerTool import DPOTrainerTool
from algorithm.llm.arguments.DPOTraingingAugments import DPOTrainingArguments
class DPOTraining(Base):
    name = "dpo_training"
    config_file = "%s%s" % (name, "_config")
    setting_file = "%s%s" % (name, "_setting")

    def __init__(self, config_file_input=None, setting_file_input=None, log=None):
        super(DPOTraining, self).__init__()
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
        print("开始运行:", DPOTraining.name)

    def after_run(self):
        print("结束运行:", DPOTraining.name)

    def run(self):
        # parse args
        model_args, data_args, training_args, script_args = self.initial_train(TraingingArguments=DPOTrainingArguments)

        # load tokenizer
        tokenizer = self.load_tokenizer(model_args=model_args)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0  # set as the <unk> token
        # 加载数据集
        dataset_class = DPOTrainingDataSet(script_args=script_args, data_args=data_args, model_args=model_args,
                                           logger=logger, training_args=training_args,
                                           tokenizer=tokenizer)
        raw_datasets = dataset_class.load_raw_data(file_suffix=["*.json", "*.jsonl"])
        train_dataset, max_train_samples, eval_dataset, max_eval_samples = dataset_class.prepare_datasets(raw_datasets)
        # 加载模型
        model_class = DPOTrainingModel(model_args=model_args, script_args=script_args, data_args=data_args,
                                       training_args=training_args, logger=logger)
        model = model_class.load_model()
        trainer_class = DPOTrainerTool(model_args=model_args, script_args=script_args, data_args=data_args,
                                  training_args=training_args, logger=logger,tokenizer=tokenizer)
        trainer_class.init_trainer(model, train_dataset, eval_dataset)
        if training_args.do_train:
            trainer_class.train(model, tokenizer, max_train_samples)
        if training_args.do_eval:
            trainer_class.evaluate(max_eval_samples)



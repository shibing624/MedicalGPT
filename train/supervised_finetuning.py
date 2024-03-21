from algorithm.llm.train.base import Base
from settings import ConfigPackage
from settings import SettingPackage
import os
from loguru import logger
from algorithm.llm.data.SupervisedFinetuningDataSet import SupervisedFinetuningDataSet
from algorithm.llm.model.SupervisedFineModel import SupervisedFineModel
from algorithm.llm.data.template.DataTemplate import get_conv_template


class Supervised_Finetuning(Base):
    name = "supervised_finetuning"
    config_file = "%s%s" % (name, "_config")
    setting_file = "%s%s" % (name, "_setting")

    def __init__(self, config_file_input=None, setting_file_input=None, log=None):
        super(Supervised_Finetuning, self).__init__()
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
        print("开始运行:", Supervised_Finetuning.name)

    def run(self):
        model_args, data_args, training_args, script_args = self.initial_train()
        tokenizer = self.define_tokenizer(data_args, model_args)
        # 加载数据集
        dataset_class = SupervisedFinetuningDataSet(script_args=script_args, data_args=data_args, model_args=model_args,
                                                    logger=logger, training_args=training_args,
                                                    tokenizer=tokenizer)
        raw_datasets = dataset_class.load_raw_data(file_suffix=["*.json", "*.jsonl"])

        train_dataset, max_train_samples, eval_dataset, max_eval_samples = dataset_class.prepare_datasets(raw_datasets)

        # 加载模型
        model_class = SupervisedFineModel(model_args=model_args, script_args=script_args, data_args=data_args,
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

    def after_run(self):
        print("结束运行:", Supervised_Finetuning.name)

    def define_tokenizer(self, data_args, model_args):
        tokenizer = self.load_tokenizer(model_args=model_args)
        prompt_template = get_conv_template(data_args.template_name)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = prompt_template.stop_str  # eos token is required for SFT
            logger.info("Add eos token: {}".format(tokenizer.eos_token))
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Add pad token: {}".format(tokenizer.pad_token))

        logger.debug(f"Tokenizer: {tokenizer}")
        return tokenizer


if __name__ == '__main__':
    model_run = Supervised_Finetuning()
    model_run.run()

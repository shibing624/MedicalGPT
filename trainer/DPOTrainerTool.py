from algorithm.llm.trainer.TrainerTool import TrainerTool
from algorithm.llm.model.Model import find_all_linear_names
from algorithm.llm.model.Model import print_trainable_parameters
from peft import LoraConfig, TaskType
from trl import DPOTrainer
from copy import deepcopy


class DPOTrainerTool(TrainerTool):

    def __init__(self, **kwargs):
        super(DPOTrainerTool, self).__init__(**kwargs)

    def init_trainer(self, model, train_dataset, eval_dataset):
        full_max_length = self.data_args.max_source_length + self.data_args.max_target_length
        self.training_args.run_name = f"dpo_{self.model_args.model_type}"
        peft_config = None
        if self.script_args.use_peft:
            self.logger.info("Fine-tuning method: LoRA(PEFT)")
            target_modules = self.script_args.target_modules.split(',') if self.script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=self.model_args.load_in_4bit,
                                                       int8=self.model_args.load_in_8bit)
            self.logger.info(f"Peft target_modules: {target_modules}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=self.script_args.lora_rank,
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=self.script_args.lora_dropout,
            )
        else:
            self.logger.info("Fine-tuning method: Full parameters training")
        self.trainer = DPOTrainer(
            model,
            ref_model=None,
            args=self.training_args,
            beta=self.training_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config if self.script_args.use_peft else None,
            max_prompt_length=self.data_args.max_source_length,
            max_length=full_max_length,
        )
        print_trainable_parameters(self.trainer.model)

    def train(self, model, tokenizer, max_train_samples):
        self.logger.info("*** Train ***")
        train_result = self.trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        if self.trainer.is_world_process_zero():
            self.logger.debug(f"Training metrics: {metrics}")
            self.logger.info(f"Saving model checkpoint to {self.training_args.output_dir}")
            self.trainer.save_model(self.training_args.output_dir)
            # tokenizer.save_pretrained(self.training_args.output_dir)
            # self.trainer.model.save_pretrained(self.training_args.output_dir)

    def evaluate(self, max_eval_samples):
        self.logger.info("*** Evaluate ***")
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        if self.trainer.is_world_process_zero():
            self.logger.debug(f"Eval metrics: {metrics}")

    # def before_init_trainer(self, model):
    #     if self.training_args.gradient_checkpointing:
    #         model.gradient_checkpointing_enable()
    #         model.config.use_cache = False
    #     else:
    #         model.config.use_cache = True

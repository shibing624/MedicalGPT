import os
import torch
import math

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled


class TrainerTool(object):

    def __init__(self, **kwargs):
        self.model_args = kwargs["model_args"]
        self.script_args = kwargs["script_args"]
        self.training_args = kwargs["training_args"]
        self.data_args = kwargs["data_args"]
        self.logger = kwargs["logger"]
        self.tokenizer = kwargs["tokenizer"]
        self.trainer = None

    def init_trainer(self, model, train_dataset, eval_dataset):
        pass

    def before_init_trainer(self, model):
        if self.training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.config.use_cache = True
        model.enable_input_require_grads()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if not ddp and torch.cuda.device_count() > 1:
            # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    def train(self, model, tokenizer, max_train_samples):
        self.logger.info("*** Train ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if self.trainer.is_world_process_zero():
            self.logger.debug(f"Training metrics: {metrics}")
            self.logger.info(f"Saving model checkpoint to {self.training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                self.save_model_zero3(model, tokenizer, self.trainer)
            else:
                self.save_model(model, tokenizer)

    def evaluate(self, max_eval_samples):

        self.logger.info("*** Evaluate ***")
        metrics = self.trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        if self.trainer.is_world_process_zero():
            self.logger.debug(f"Eval metrics: {metrics}")

    def save_model_zero3(self, model, tokenizer, trainer):
        output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(self.training_args.output_dir, state_dict=state_dict_zero3)
        tokenizer.save_pretrained(output_dir)

    def save_model(self, model, tokenizer):
        """Save the model and the tokenizer."""
        output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

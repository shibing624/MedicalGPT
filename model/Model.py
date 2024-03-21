import math
import os

import numpy as np
import torch

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    Trainer,
    is_torch_tpu_available,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers.trainer import TRAINING_ARGS_NAME
from typing import List, Dict, Any, Mapping
from sklearn.metrics import accuracy_score

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics, we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


class Model(object):

    def __init__(self, **kwargs):
        self.model_args = kwargs["model_args"]
        self.script_args = kwargs["script_args"]
        self.training_args = kwargs["training_args"]
        self.data_args = kwargs["data_args"]
        self.logger = kwargs["logger"]
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_args.model_type]
        self.trainer = None
        set_seed(self.training_args.seed)

    def before_load_model(self):
        # Load model
        if self.model_args.model_type and self.model_args.model_name_or_path:
            torch_dtype = (
                self.model_args.torch_dtype
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            ddp = world_size != 1
            if ddp:
                self.model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            if self.script_args.qlora and (len(self.training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
                self.logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")

            config = self.config_class.from_pretrained(
                self.model_args.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.model_args.trust_remote_code,
                cache_dir=self.model_args.cache_dir
            )
            return config, torch_dtype
        else:
            raise ValueError(
                f"Error, model_name_or_path is None, Continue PT/SFT must be loaded from a pre-trained model")

    def load_model(self):
        config, torch_dtype = self.before_load_model()
        load_in_4bit = self.model_args.load_in_4bit
        load_in_8bit = self.model_args.load_in_8bit
        load_in_8bit_skip_modules = None
        if load_in_8bit or load_in_4bit:
            self.logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if self.script_args.modules_to_save is not None:
                load_in_8bit_skip_modules = self.script_args.modules_to_save.split(',')
        model = self.model_class.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=self.model_args.device_map,
            trust_remote_code=self.model_args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                load_in_8bit_skip_modules=load_in_8bit_skip_modules,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            ) if self.script_args.qlora else None,
        )
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        return self.after_load_model(model, config)

    def after_load_model(self, model, config):
        return model

    def before_fine_tuning_model(self, model):
        return model

    def fine_tuning_model(self, model, tokenizer,task_type=TaskType.CAUSAL_LM):
        model = self.before_fine_tuning_model(model)
        self.logger.info("Fine-tuning method: LoRA(PEFT)")
        if self.script_args.peft_path is not None:
            self.logger.info(f"Peft from pre-trained model: {self.script_args.peft_path}")
            model = PeftModel.from_pretrained(model, self.script_args.peft_path, is_trainable=True)
        else:
            self.logger.info("Init new peft model")
            load_in_4bit = self.model_args.load_in_4bit
            load_in_8bit = self.model_args.load_in_8bit
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, self.training_args.gradient_checkpointing)
            target_modules = self.script_args.target_modules.split(',') if self.script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = self.script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
                # Resize the embedding layer to match the new tokenizer
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"Peft target_modules: {target_modules}")
            self.logger.info(f"Peft lora_rank: {self.script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=task_type,
                target_modules=target_modules,
                inference_mode=False,
                r=self.script_args.lora_rank,
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=self.script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
        return self.after_before_fine_tuning_model(model)

    def after_before_fine_tuning_model(self, model):
        return model

    def full_training_model(self, model):
        self.logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)
        return model

    def before_initial_trainer(self, model):
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

    def initial_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        # Initialize our Trainer
        if self.trainer:
            return
        self.before_initial_trainer(model)
        self.trainer = SavePeftModelTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=fault_tolerance_data_collator,
            compute_metrics=compute_metrics if self.training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if self.training_args.do_eval and not is_torch_tpu_available()
            else None,
        )

    def before_train(self, tokenizer):
        self.logger.debug(f"Train dataloader example: {next(iter(self.trainer.get_train_dataloader()))}")

    def train(self, model, tokenizer, train_dataset, eval_dataset, max_train_samples):
        self.initial_trainer(model, tokenizer, train_dataset, eval_dataset)
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

    def evaluate(self, model, tokenizer, train_dataset, eval_dataset, max_eval_samples):
        self.initial_trainer(model, tokenizer, train_dataset, eval_dataset)
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

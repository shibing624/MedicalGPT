import os
from typing import List, Optional, Dict, Union, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import Trainer, PreTrainedTokenizerBase
from transformers.trainer import TRAINING_ARGS_NAME
from sklearn.metrics import mean_squared_error, mean_absolute_error
from algorithm.llm.model.Model import Model


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Here, predictions is rewards_chosen and rewards_rejected.
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # MSE
    mse = mean_squared_error(labels, preds)
    # MAE
    mae = mean_absolute_error(labels, preds)

    return {"mse": mse, "mae": mae}


@dataclass
class RewardDataCollatorWithPadding:
    """We need to define a special data collator that batches the data in our chosen vs rejected format"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prepare inputs for chosen and rejected separately
        device = model.device

        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }
        outputs_chosen = model(**inputs_chosen)
        rewards_chosen = outputs_chosen.logits.detach()

        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }
        outputs_rejected = model(**inputs_rejected)
        rewards_rejected = outputs_rejected.logits.detach()

        # Keep the compute_loss method
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


class RewardModel(Model):
    def __init__(self, **kwargs):
        super(RewardModel, self).__init__(**kwargs)

    def load_model(self):
        config, torch_dtype = self.before_load_model()
        if self.model_args.model_type in ['bloom', 'llama']:
            model = self.model_class.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                load_in_4bit=self.model_args.load_in_4bit,
                load_in_8bit=self.model_args.load_in_8bit,
                device_map=self.model_args.device_map,
                trust_remote_code=self.model_args.trust_remote_code,
            )
        else:
            model = self.model_class.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
                cache_dir=self.model_args.cache_dir,
                ignore_mismatched_sizes=True
            )
            model.to(self.training_args.device)
        return model

    def initial_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        if self.trainer:
            return
        full_max_length = self.data_args.max_source_length + self.data_args.max_target_length
        self.before_initial_trainer(model)
        self.trainer = RewardTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=RewardDataCollatorWithPadding(
                tokenizer=tokenizer, max_length=full_max_length, padding="max_length"
            ),
        )

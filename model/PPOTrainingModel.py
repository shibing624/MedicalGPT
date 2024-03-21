import os

import torch
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from algorithm.llm.model.Model import Model
from algorithm.llm.model.Model import print_trainable_parameters


def get_reward_model_output(reward_model, reward_tokenizer, question, answer, device):
    """
    Get the reward score for a given question and answer pair.
    """
    inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
    score = reward_model(**inputs).logits[0].cpu().detach()

    return score


def calculate_rewards(reward_score_outputs, reward_baseline=0):
    """
    Calculate the reward for a given score output.
    :param reward_score_outputs:
    :param reward_baseline:
    :return:
    """
    rewards = []
    for score in reward_score_outputs:
        if isinstance(score, torch.Tensor) and score.numel() == 1:
            reward_value = score.item() - reward_baseline
            rewards.append(torch.tensor(reward_value))
        else:
            # Use the average of the tensor elements as `score` is multiple elements
            reward_value = torch.mean(score).item() - reward_baseline
            rewards.append(torch.tensor(reward_value))
    return rewards


class PPOTrainingModel(Model):

    def __init__(self, **kwargs):
        super(PPOTrainingModel, self).__init__(**kwargs)
        self.config = None
        self.device = None
        self.reward_model = None

    def load_model(self):

        peft_config = None
        if self.script_args.use_peft:
            self.logger.info("Fine-tuning method: LoRA(PEFT)")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=None if self.script_args.target_modules == 'all' else self.script_args.target_modules,
                inference_mode=False,
                r=self.script_args.lora_rank,
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=self.script_args.lora_dropout,
            )
        else:
            self.logger.info("Fine-tuning method: Full parameters training")
        config, torch_dtype = self.before_load_model()
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            load_in_4bit=self.model_args.load_in_4bit,
            load_in_8bit=self.model_args.load_in_8bit,
            device_map=self.model_args.device_map,
            trust_remote_code=self.model_args.trust_remote_code,
            peft_config=peft_config if self.script_args.use_peft else None,
        )
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        print_trainable_parameters(model)
        # Load reward model
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.training_args.reward_model_device if self.training_args.reward_model_device is not None else default_device
        reward_config = self.config_class.from_pretrained(
            self.model_args.reward_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.model_args.trust_remote_code,
            cache_dir=self.model_args.cache_dir
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.reward_model_name_or_path,
            config=reward_config,
            load_in_8bit=self.model_args.load_in_8bit,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        self.reward_model.to(self.device)
        tokenizer_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "use_fast": self.model_args.use_fast_tokenizer,
            "trust_remote_code": self.model_args.trust_remote_code,
        }
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.reward_model_name_or_path, **tokenizer_kwargs
        )
        return model

    def initial_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        if self.trainer:
            return
        self.config = PPOConfig(
            steps=self.training_args.max_steps,
            model_name=self.model_args.model_name_or_path,
            learning_rate=self.training_args.learning_rate,
            log_with="tensorboard",
            batch_size=self.training_args.batch_size,
            mini_batch_size=self.training_args.mini_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=self.training_args.early_stopping,
            target_kl=self.training_args.target_kl,
            seed=self.training_args.seed,
            init_kl_coef=self.training_args.init_kl_coef,
            adap_kl_ctrl=self.training_args.adap_kl_ctrl,
            project_kwargs={"logging_dir": self.training_args.output_dir},
        )

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        # full_max_length = self.data_args.max_source_length + self.data_args.max_target_length
        # self.before_initial_trainer(model)
        print("test config:{}".format(self.config.to_dict()))
        self.trainer = PPOTrainer(
            config=self.config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=train_dataset,
            data_collator=collator,
        )

    def train(self, model, tokenizer, train_dataset, eval_dataset, max_train_samples):
        # These arguments are passed to the `generate` function of the PPOTrainer
        generation_kwargs = {
            "max_new_tokens": self.data_args.max_target_length,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        self.initial_trainer(model, tokenizer, train_dataset, eval_dataset)
        self.logger.info("*** Train ***")
        total_steps = self.config.total_ppo_epochs
        for step, batch in tqdm(enumerate(self.trainer.dataloader)):
            if step >= total_steps:
                break
            question_tensors = batch["input_ids"]
            question_tensors = [torch.LongTensor(i).to(self.device).squeeze(0) for i in question_tensors]
            responses = []
            response_tensors = []
            for q_tensor in question_tensors:
                response_tensor = self.trainer.generate(
                    q_tensor,
                    return_prompt=False,
                    **generation_kwargs,
                )
                r = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)[0]
                responses.append(r)
                response_tensors.append(response_tensor.squeeze(0))
            batch["response"] = responses

            # Compute reward score
            score_outputs = [
                get_reward_model_output(self.reward_model, self.reward_tokenizer, q, r, self.device) for q, r in
                zip(batch["query"], batch["response"])
            ]
            rewards = calculate_rewards(score_outputs, self.training_args.reward_baseline)

            # Run PPO step
            try:
                stats = self.trainer.step(question_tensors, response_tensors, rewards)
                self.trainer.log_stats(stats, batch, rewards)
                self.logger.debug(f"Step {step}/{total_steps}: reward score:{score_outputs}")
            except ValueError as e:
                self.logger.warning(f"Failed to log stats for step {step}, because of {e}")

            if step and step % self.training_args.save_steps == 0:
                save_dir = os.path.join(self.training_args.output_dir, f"checkpoint-{step}")
                self.trainer.save_pretrained(save_dir)
        # Save final model
        self.trainer.save_pretrained(self.training_args.output_dir)

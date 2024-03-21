from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments
@dataclass
class PPOTrainingArguments(Seq2SeqTrainingArguments):
    ## ppo
    reward_model_device: Optional[str] = field(default="cuda:0", metadata={"help": "The reward model device"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "Whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0, metadata={"help": "Baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # learning_rate: Optional[float] = field(default=1.5e-5, metadata={"help": "Learning rate"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    output_dir: Optional[str] = field(default="outputs-rl", metadata={"help": "The output directory"})
    seed: Optional[int] = field(default=0, metadata={"help": "Seed"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    learning_rate: Optional[float] = field(default=1.5e-5, metadata={"help": "Learning rate"})
    # report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})
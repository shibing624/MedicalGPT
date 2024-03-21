from dataclasses import dataclass, field
from typing import Optional
@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )

    # ## ppo
    # batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size"})
    # early_stopping: Optional[bool] = field(default=False, metadata={"help": "Whether to early stop"})
    # target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    # reward_baseline: Optional[float] = field(
    #     default=0.0, metadata={"help": "Baseline value that is subtracted from the reward"},
    # )
    # init_kl_coef: Optional[float] = field(
    #     default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    # )
    # adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # # learning_rate: Optional[float] = field(default=1.5e-5, metadata={"help": "Learning rate"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=1, metadata={"help": "the number of gradient accumulation steps"}
    # )
    # save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    # output_dir: Optional[str] = field(default="outputs-rl", metadata={"help": "The output directory"})
    # seed: Optional[int] = field(default=0, metadata={"help": "Seed"})
    # max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    # report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")

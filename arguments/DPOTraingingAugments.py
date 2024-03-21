from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments
@dataclass
class DPOTrainingArguments(Seq2SeqTrainingArguments):
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
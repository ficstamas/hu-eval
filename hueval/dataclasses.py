from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase, DataCollator
from typing import Optional, Callable
from datasets import DatasetDict
from evaluate import EvaluationModule


@dataclass
class RunParameters:
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    dataset: Optional[DatasetDict] = None
    tokenized_dataset: Optional[DatasetDict] = None
    metric: Optional[EvaluationModule] = None
    data_collator: Optional[DataCollator] = None
    compute_metrics: Optional[Callable] = None

from transformers import Trainer, TrainingArguments, IntervalStrategy, DataCollatorForTokenClassification, \
    DataCollatorWithPadding
from hueval.datasets import load_dataset, TaskType
from hueval.transformers.token_classification import create_model as token_classification_model
from hueval.transformers.sequence_classification import create_model as sequence_classification_model
from hueval.transformers.multiple_choice_qa import create_model as multiple_choice_qa_model
from hueval.tokenizers.utils.align_labels_for_token_classification import AlignLabels
from hueval.tokenizers.utils.hulu_tokenizer import SequenceTokenizer, MultipleChoiceTokenizer
from hueval.utils.data_collator import DataCollatorForMultipleChoice
import numpy as np


_PREDEFINED_TRAINING_ARGUMENTS = {
    "output_dir": "~/temp/",
    "per_device_eval_batch_size": 8,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "evaluation_strategy": IntervalStrategy.EPOCH,
    "save_strategy": IntervalStrategy.NO,
    "data_seed": 0
}


class Training:
    def __init__(self, model_name: str, task_name: str, task_configuration: str, label_name: str,
                 max_seq_length: int = 256, seed: int = 0, **training_arguments):
        dataset = load_dataset(task_name, task_configuration)
        if dataset.type == TaskType.TOKEN_CLASSIFICATION:
            params = token_classification_model(model_name, label_name, dataset)
            if label_name == 'upos':
                params.dataset = params.dataset.filter(lambda example: example['morph_tagged'])
            aligner = AlignLabels(params.tokenizer, label_name, max_length=max_seq_length)
            params.data_collator = DataCollatorForTokenClassification(
                tokenizer=params.tokenizer, padding='max_length', max_length=max_seq_length
            )
            params.compute_metrics = self.compute_metrics_
        elif dataset.type == TaskType.SEQUENCE_CLASSIFICATION:
            params = sequence_classification_model(model_name, label_name, dataset)
            aligner = SequenceTokenizer(params.tokenizer, task_configuration, max_length=max_seq_length)
            params.data_collator = DataCollatorWithPadding(
                tokenizer=params.tokenizer, padding='max_length', max_length=max_seq_length
            )
            params.compute_metrics = self.compute_metrics_
        elif dataset.type == TaskType.MULTIPLE_CHOICE_QUESTION_ANSWERING:
            params = multiple_choice_qa_model(model_name, label_name, dataset)
            aligner = MultipleChoiceTokenizer(params.tokenizer, task_configuration, max_length=max_seq_length)
            params.data_collator = DataCollatorForMultipleChoice(
                tokenizer=params.tokenizer, padding='max_length', max_length=max_seq_length
            )
            params.compute_metrics = self.compute_metrics_
        else:
            raise NotImplementedError

        params.tokenized_dataset = params.dataset.map(aligner.preprocess_function, batched=True, batch_size=8)
        self.params = params
        training_arguments['seed'] = seed
        self.arguments = TrainingArguments(
            **training_arguments
        )
        self.trainer = Trainer(
            model=params.model,
            args=self.arguments,
            data_collator=params.data_collator,
            train_dataset=params.tokenized_dataset['train'],
            eval_dataset=params.tokenized_dataset['validation'],
            tokenizer=params.tokenizer,
            compute_metrics=params.compute_metrics
        )
        self.label_name = label_name

    def train(self):
        self.trainer.train()

    def eval(self):
        if self.params.dataset['test'][self.label_name][0] == -1:
            return
        self.trainer.evaluate(self.params.tokenized_dataset['test'], metric_key_prefix="test")

    def compute_metrics_(self, predictions):
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        results = self.params.metric.compute(predictions=preds, references=labels)
        return results

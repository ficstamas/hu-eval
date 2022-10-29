from transformers import AutoModelForMultipleChoice, BertForMultipleChoice, AutoTokenizer
from hueval.datasets import Dataset
from hueval.models.hubert import load_hubert
from hueval.tokenizers.hubert import load_tokenizer
from hueval.dataclasses import RunParameters


def create_model(name: str, label_column: str, dataset: Dataset) -> RunParameters:
    config = {
        'num_labels': len(dataset.dataset['train'].features[label_column].names)
    }
    if name.startswith("hubert-wiki"):
        model_type = 'cased'
        if name.endswith("uncased"):
            model_type = 'uncased'
        model = load_hubert(model_type, BertForMultipleChoice, config)
        tokenizer = load_tokenizer(model_type)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(name, **config)
        tokenizer = AutoTokenizer.from_pretrained(name)

    return RunParameters(model=model, tokenizer=tokenizer, dataset=dataset.dataset, metric=dataset.metric)
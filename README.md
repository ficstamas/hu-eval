# Hungarian Evaluation library for NLP


# Installation

Latest version:
```
pip install git+https://github.com/ficstamas/hu-eval.git
```

Specified version:
```
pip install git+https://github.com/ficstamas/hu-eval.git@v0.1.0
```

# Hubert Wiki

Loading the model:
```python
from transformers import BertForSequenceClassification
from hueval.models.hubert import load_hubert


model = load_hubert("uncased", BertForSequenceClassification)  # or "cased"
```
Loading the corresponding tokenizer:
```python
from hueval.tokenizers.hubert import load_tokenizer

tokenizer = load_tokenizer("uncased")  # or "cased"
tokenizer('Az alma leesett a fáról, egyenesen Péter fejére.')
```
The above example returns a wrapped tokenizer where ONLY the `__call__` method was wrapped, in order to call `.lower()` on
every input text. You can obtain the normal `BertTokenizer` by calling `load_tokenizer("uncased", return_wrapped=False)`.
However, you have to manually ensure that the input text is lower cased.



# Dataset Usage

```python
from hueval.datasets import load_dataset

rc_dataset = load_dataset("hulu", "rc")

# Output:
# Dataset(dataset=DatasetDict({
#     train: Dataset({
#         features: ['query', 'lead', 'passage', 'label', 'idx'],
#         num_rows: 64614
#     })
#     validation: Dataset({
#         features: ['query', 'lead', 'passage', 'label', 'idx'],
#         num_rows: 8000
#     })
#     test: Dataset({
#         features: ['query', 'lead', 'passage', 'label', 'idx'],
#         num_rows: 8000
#     })
# }), metric=EvaluationModule(...)}
```

# Supported Datasets

| Names        | Available Configurations             | Sources                                                         |
|--------------|--------------------------------------|-----------------------------------------------------------------|
| hulu         | cola, copa, sst2, rc, wnli, ws       | [link](https://github.com/nytud/HuLU)                           |
| nytk-nerkor  | fiction, legal, news, web, wikipedia | [link](https://github.com/nytud/NYTK-NerKor)                    |
| nerkor_1.41e | fiction, legal, news, web, wikipedia | [link](https://github.com/novakat/NYTK-NerKor-Cars-OntoNotesPP) | 
| opinhubank   | opinhubank                           | [link](https://sites.google.com/site/mmihaltz/resources)        | 

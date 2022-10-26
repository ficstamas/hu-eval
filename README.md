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

# Usage

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

| Names       | Available Configurations             |
|-------------|--------------------------------------|
| hulu        | cola, copa, sst2, rc, wnli, ws       |
| nytk-nerkor | fiction, legal, news, web, wikipedia | 

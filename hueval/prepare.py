from hueval.datasets import load_dataset, available_datasets, available_configs
from hueval.models.hubert import load_hubert
from itertools import product


all_models = [
    "SzegedAI/hubert-tiny-wiki-seq128", "SzegedAI/hubert-tiny-wiki", "SzegedAI/hubert-small-wiki-seq128",
    "SzegedAI/hubert-small-wiki", "SzegedAI/hubert-medium-wiki-seq128", "SzegedAI/hubert-medium-wiki",
    "hubert-wiki-cased", "hubert-wiki-uncased", "SZTAKI-HLT/hubert-base-cc",
    "xlm-roberta-base", "xlm-roberta-large", "bert-base-multilingual-cased", "distilbert-base-multilingual-cased",
    "google/rembert"
]

all_tasks = [
    ('hulu', 'cola', 'labels'),
    ('hulu', 'sst2', 'labels'),
    ('hulu', 'wnli', 'labels'),
    ('hulu', 'copa', 'labels'),
    ('hulu', 'ws', 'labels'),
    ('nytk-nerkor', 'fiction', 'ner'),
    ('nytk-nerkor', 'legal', 'ner'),
    ('nytk-nerkor', 'news', 'upos'),
    ('nytk-nerkor', 'news', 'ner'),
    ('nytk-nerkor', 'web', 'upos'),
    ('nytk-nerkor', 'web', 'ner'),
    ('nytk-nerkor', 'wikipedia', 'upos'),
    ('nytk-nerkor', 'wikipedia', 'ner'),
    ('nerkor_1.41e', 'fiction', 'ner'),
    ('nerkor_1.41e', 'legal', 'ner'),
    ('nerkor_1.41e', 'news', 'ner'),
    ('nerkor_1.41e', 'web', 'ner'),
    ('nerkor_1.41e', 'wikipedia', 'ner'),
    ('opinhubank', 'opinhubank', 'labels')
]

all_configurations = [
    (x, y[0], y[1], y[2]) for x, y in product(all_models, all_tasks)
]

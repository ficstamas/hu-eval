import torch
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
import os
from hueval.utils.network import download_url
from typing import Literal, Union, Type
from transformers import BertForSequenceClassification, BertForPreTraining, BertForTokenClassification, \
    BertForMultipleChoice, BertForMaskedLM, BertForQuestionAnswering, BertForNextSentencePrediction, BertConfig
import tarfile
import json
from pathlib import Path


_MODELS = Union[
    Type[BertForSequenceClassification],
    Type[BertForPreTraining],
    Type[BertForTokenClassification],
    Type[BertForMultipleChoice],
    Type[BertForMaskedLM],
    Type[BertForQuestionAnswering],
    Type[BertForNextSentencePrediction]
]
_RETURN_MODELS = Union[
    BertForSequenceClassification,
    BertForPreTraining,
    BertForTokenClassification,
    BertForMultipleChoice,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForNextSentencePrediction
]
_TYPES = Literal['cased', 'uncased']
_HUBERT_WIKI = {
    "cased": "https://nessie.ilab.sztaki.hu/~ndavid/hubert/hubert_wiki.tar.gz",
    "uncased": "https://nessie.ilab.sztaki.hu/~ndavid/hubert/hubert_wiki_lower.tar.gz"
}


def download_model(model_type: _TYPES):
    link = _HUBERT_WIKI[model_type]
    if 'TRANSFORMERS_CACHE' in os.environ:
        cache_dir = os.environ['TRANSFORMERS_CACHE']
    else:
        home = str(Path.home())
        cache_dir = os.path.join(home, ".cache/huggingface/transformers/")

    download_path = os.path.join(cache_dir, f"hubert-wiki-{model_type}/hubert-wiki-{model_type}.tar.gz")
    if os.path.exists(download_path):
        return download_path

    download_url(link, download_path)
    return download_path


def extract(path: str):
    with tarfile.open(path) as f:
        destination = path.rstrip(".tar.gz")
        if not os.path.exists(destination):
            f.extractall(destination)
    return destination


def download_and_extract(model_type: _TYPES):
    path = download_model(model_type)
    path = extract(path)

    # appending model_type to config
    # the lack of this argument affects the behaviour of tokenizer
    prefix = "hubert_wiki" if model_type == "cased" else "hubert_wiki_lower"
    with open(os.path.join(path, prefix, "config.json"), mode="r") as f:
        config = json.load(f)
    if "model_type" not in config:
        config["model_type"] = "bert"
        with open(os.path.join(path, prefix, "config.json"), mode="w") as f:
            json.dump(config, f)
    return path


def convert_model(model_type: _TYPES):
    path = download_and_extract(model_type)
    prefix = "hubert_wiki" if model_type == "cased" else "hubert_wiki_lower"
    torch_model_path = os.path.join(
        path, f"{prefix}/pytorch_model.bin"
    )
    config_path = os.path.join(path, f"{prefix}/config.json")
    if not os.path.exists(torch_model_path):
        convert_tf_checkpoint_to_pytorch(
            tf_checkpoint_path=os.path.join(
                path, f"{prefix}/model.ckpt-100000.index"
            ),
            bert_config_file=config_path,
            pytorch_dump_path=torch_model_path
        )
    return torch_model_path, config_path


def load_hubert(model_type: _TYPES, model_class: _MODELS) -> _RETURN_MODELS:
    """
    Loads Hubert wiki weights into the provided model which is equivalent to a 'SZTAKI-HLT/hubert-base-cc' in terms of
    parameters
    :param model_type: cased, uncased
    :param model_class: Class of the model (like `transformer.BertForSequenceClassification`) NOT the initialized object
    :return: desired model with the appropriate weights
    """
    path, cfg = convert_model(model_type)
    with open(cfg, mode='r') as f:
        config = json.load(f)
    model = model_class(BertConfig(**config))
    model.load_state_dict(torch.load(path), strict=False)
    return model


from .hulu import Hulu
from collections import namedtuple
from evaluate import load
from typing import List
from .nerkor import _SUBS as NERKOR_SUBS
from .nerkor import NerKor
from .nerkor_extended import NerKorExtended
from .opinhubank import OpinHuBank
from enum import Enum


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 0
    SEQUENCE_CLASSIFICATION = 1
    SPAN_CLASSIFICATION = 2
    MULTIPLE_CHOICE_QUESTION_ANSWERING = 3


Dataset = namedtuple("Dataset", ['dataset', 'metric', 'type'])



_metric = {
    "hulu": {
        "cola": ["glue", "cola"],
        "sst2": ["glue", "sst2"],
        "wnli": ["glue", "wnli"],
        "rc": ["super_glue", "record"],
        "copa": ["super_glue", "copa"],
        "ws": ["super_glue", "wsc"],
    },
    "nytk-nerkor": {x: ["seqeval"] for x in NERKOR_SUBS},
    "nerkor_1.41e": {x: ["seqeval"] for x in NERKOR_SUBS},
    "opinhubank": {
        "opinhubank": ["accuracy"]
    },
}


def available_configs(name: str) -> List[str]:
    """
    Returns available configurations for a given dataset
    :param name: Name of the Dataset
    :return:
    """
    return [x for x in _metric[name].keys()]


def available_datasets() -> List[str]:
    return [x for x in _metric.keys()]


def load_dataset(name: str, config: str) -> Dataset:
    """
    Returns a Dataset with its corresponding metric
    :param name: Name of the dataset
    :param config: Name of the dataset configuration
    :return:
    """
    if name == "hulu":
        builder = Hulu(config_name=config)
        dataset = builder.download_and_prepare()
        dataset = builder.as_dataset()
        task_type = TaskType.SEQUENCE_CLASSIFICATION
        if config == "rc":
            task_type = TaskType.SPAN_CLASSIFICATION
        elif config in ["ws", "copa"]:
            task_type = TaskType.MULTIPLE_CHOICE_QUESTION_ANSWERING
        return Dataset(dataset=dataset, metric=load(*_metric[name][config]),
                       type=task_type)
    elif name == "nytk-nerkor":
        builder = NerKor(config_name=config)
        dataset = builder.download_and_prepare()
        dataset = builder.as_dataset()
        return Dataset(dataset=dataset, metric=load(*_metric[name][config]), type=TaskType.TOKEN_CLASSIFICATION)
    elif name == "nerkor_1.41e":
        builder = NerKorExtended(config_name=config)
        dataset = builder.download_and_prepare()
        dataset = builder.as_dataset()
        return Dataset(dataset=dataset, metric=load(*_metric[name][config]), type=TaskType.TOKEN_CLASSIFICATION)
    elif name == "opinhubank":
        builder = OpinHuBank(config_name=config)
        dataset = builder.download_and_prepare()
        dataset = builder.as_dataset()
        return Dataset(dataset=dataset, metric=load(*_metric[name][config]), type=TaskType.SEQUENCE_CLASSIFICATION)
    raise NotImplementedError(f"Dataset {name} does not exists")


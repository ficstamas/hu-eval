from .hulu import Hulu
from collections import namedtuple
from evaluate import load
from typing import List


Dataset = namedtuple("Dataset", ['dataset', 'metric'])
_metric = {
    "hulu": {
        "cola": ["glue", "cola"],
        "sst2": ["glue", "sst2"],
        "wnli": ["glue", "wnli"],
        "rc": ["super_glue", "record"],
        "copa": ["super_glue", "copa"],
        "ws": ["super_glue", "wsc"],
    }
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
        return Dataset(dataset=dataset, metric=load(*_metric[name][config]))
    raise NotImplementedError(f"Dataset {name} does not exists")

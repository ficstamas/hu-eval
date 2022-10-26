from datasets import GeneratorBasedBuilder, BuilderConfig, Version, DatasetInfo, Features, Value, \
    Sequence, ClassLabel, DownloadManager, SplitGenerator, Split
import os
import textwrap
from hueval.utils.network import download_url
import csv
import numpy as np
from sklearn.model_selection import train_test_split


_CITATION = """
Miháltz, Márton (2013). “OpinHuBank: szabadon hozzáférhető annotált korpusz magyar nyelvű véleményelemzéshez”.
Tanács Attila, Vincze Veronika (szerk.): IX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2013), SZTE, Szeged, 2013, pp. 343-345.
"""

_SOURCE = "https://metashare.nytud.hu/repository/download/608756be64e211e2aa7c68b599c26a068dd5b3551f024f6281131670412d37d3/"


class OpinHuBankConfig(BuilderConfig):
    """BuilderConfig for OpinHuBankConfig."""

    def __init__(
            self,
            text_features,
            int_features,
            label_classes,
            data_url,
            data_dir,
            url,
            **kwargs,
    ):
        super(OpinHuBankConfig, self).__init__(version=Version("1.0.0", ""), **kwargs)
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.text_features = text_features
        self.int_features = int_features
        self.url = url


class OpinHuBank(GeneratorBasedBuilder):
    """Hulu datasets."""

    BUILDER_CONFIGS = [
        OpinHuBankConfig(
            name="opinhubank",
            description=textwrap.dedent(""""""),
            text_features=["entity", "sentence", "url"],
            int_features=["annotators", "start", "len"],
            label_classes=["negative", "neutral", "positive"],
            data_url=_SOURCE,
            data_dir="opinhubank",
            url="https://sites.google.com/site/mmihaltz/resources"
        )
    ]

    def _info(self):
        l_ = ClassLabel(names=self.config.label_classes)
        features = {text_feature: Value("string") for text_feature in self.config.text_features}
        features["annotators"] = Sequence(l_)
        features["start"] = Value("int32")
        features["len"] = Value("int32")
        features["label"] = l_
        features["idx"] = Value("int32")
        return DatasetInfo(
            description=self.config.description,
            features=Features(features),
            homepage=self.config.url,
            citation=textwrap.dedent(_CITATION),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        path = dl_manager.download_custom(self.config.data_url, self._custom_download)
        path = dl_manager.extract(path)
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"split_key": "train", "data_file": path},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"split_key": "validation", "data_file": path},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"split_key": "test", "data_file": path},
            )
        ]

    @staticmethod
    def _custom_download(url, path):
        download_url(url, path, 5, False, {'desc': "opinhubank.zip"}, "post", {
            "licence_agree": "on",
            "in_licence_agree_form": "True",
            "licence": "CC-BY"
        })

    def _generate_examples(self, data_file, split_key, **kwargs):
        path = os.path.join(data_file, "OpinHuBank_20130106.csv")
        data = []
        with open(path, mode="r", encoding="iso-8859-2") as f:
            content = csv.reader(f)
            skip = True
            for i, line in enumerate(content):
                if skip:
                    skip = False
                    continue
                labels = [int(line[x])+1 for x in range(6, 11)]
                uniq = np.unique(labels, return_counts=True)
                majority_label_index = np.argmax(uniq[1])
                majority_label = uniq[0][majority_label_index]
                if uniq[1][majority_label_index] == 2:
                    majority_label = 1
                data.append({
                    "idx": i,
                    "start": line[1],
                    "len": line[2],
                    "entity": line[3],
                    "sentence": line[4],
                    "url": line[5],
                    "annotators": labels,
                    "label": majority_label
                })
        train, test = train_test_split(data, train_size=0.8, random_state=0)
        train, validation = train_test_split(train, train_size=int(len(data)*0.7), random_state=0)
        if split_key == "train":
            data = train
        elif split_key == "validation":
            data = validation
        elif split_key == "test":
            data = test
        for n, d in enumerate(data):
            yield n, d

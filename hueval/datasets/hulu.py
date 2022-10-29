from datasets import GeneratorBasedBuilder, BuilderConfig, Version, DatasetInfo, Features, Value, \
    Sequence, ClassLabel, DownloadManager, SplitGenerator, Split
import os
import json
import textwrap
from sklearn.model_selection import train_test_split


_CITATION = """"""


_SOURCES = {
    "cola": "https://github.com/nytud/HuCOLA/archive/32752a757dbecba7c935e6d75641758aeccbfd54.zip",
    "copa": "https://github.com/nytud/HuCoPA/archive/088bcf06ea16bc62fe4ee0cdbf083d3209236a4c.zip",
    "wnli": "https://github.com/nytud/HuWNLI/archive/6c79db979d0053511d986c15e0da784f1e33c0eb.zip",
    "sst2": "https://github.com/nytud/HuSST/archive/bcf352f37ddd4c5257245adea5defeaf8d1bc148.zip",
    "rc": {
        "train": "https://huggingface.co/datasets/NYTK/HuRC/resolve/b45ea6dbdec3b8692f02df89e8f943fa8d84e5bf/data/hurc_train.json",
        "test": "https://huggingface.co/datasets/NYTK/HuRC/resolve/b45ea6dbdec3b8692f02df89e8f943fa8d84e5bf/data/hurc_test.json",
        "validation": "https://huggingface.co/datasets/NYTK/HuRC/resolve/b45ea6dbdec3b8692f02df89e8f943fa8d84e5bf/data/hurc_val.json"
    },
    "ws": "https://github.com/nytud/HuWS/archive/cd12288254f0e5db9976eca6bcf1184c521eef94.zip"
}

_PATHS = {
    "cola": {"train": "data/cola_train.json", "validation": "data/cola_dev.json", "test": "data/cola_test.json"},
    "copa": {"train": "data/train.json", "validation": "data/val.json", "test": "data/test.json"},
    "wnli": {"train": "data/train.json", "validation": "data/dev.json", "test": "data/test.json"},
    "sst2": {"train": "data/sst_train.json", "validation": "data/sst_dev.json", "test": "data/sst_test.json"},
    "rc": {"train": "data/hurc_train.json", "validation": "data/hurc_val.json", "test": "data/hurc_test.json"},
    "ws": {"train": "huws.json", "validation": "huws.json", "test": "huws.json"}
}


def get_repo_url(x):
    if x.startswith("https://huggingface"):
        return "/".join(x.split("/")[:6])
    return "/".join(x.split("/")[:5])


class HuluConfig(BuilderConfig):
    """BuilderConfig for Hulu."""

    def __init__(
            self,
            text_features,
            label_classes,
            label_column,
            data_url,
            data_dir,
            url,
            **kwargs,
    ):
        super(HuluConfig, self).__init__(version=Version("1.0.0", ""), **kwargs)
        self.label_column = label_column
        self.data_url = data_url
        self.data_dir = data_dir
        self.text_features = text_features
        self.label_classes = label_classes
        self.url = url


class Hulu(GeneratorBasedBuilder):
    """Hulu datasets."""

    BUILDER_CONFIGS = [
        HuluConfig(
            name="cola",
            description=textwrap.dedent(""""""),
            text_features={"sentence": "sentence"},
            label_classes=["unacceptable", "acceptable"],
            label_column="Label",
            data_url=_SOURCES["cola"],
            data_dir="hulu/Cola",
            url=get_repo_url(_SOURCES["cola"])
        ),
        HuluConfig(
            name="copa",
            description="",
            text_features={"premise": "premise",
                           "choice1": "choice1",
                           "choice2": "choice2",
                           "question": "question"},
            label_classes=["choice1", "choice2"],
            label_column="label",
            data_url=_SOURCES["copa"],
            data_dir="hulu/Copa",
            url=get_repo_url(_SOURCES["copa"])
        ),
        HuluConfig(
            name="wnli",
            description=textwrap.dedent(""""""),
            text_features={"sentence1": "sentence1",
                           "sentence2": "sentence2"},
            label_classes=["not_entailment", "entailment"],
            label_column="label",
            data_url=_SOURCES["wnli"],
            data_dir="hulu/wnli",
            url=get_repo_url(_SOURCES["wnli"])
        ),
        HuluConfig(
            name="rc",
            description=textwrap.dedent(""""""),
            text_features={"lead": "lead", "passage": "passage", "query": "query"},
            label_classes=["mask"],
            label_column="answer",
            data_url=_SOURCES["rc"],
            data_dir="hulu/rc",
            url=get_repo_url(_SOURCES["rc"]['train'])
        ),
        HuluConfig(
            name="ws",
            description=textwrap.dedent(""""""),
            text_features={"sentence": "sentence",
                           "choice1": "choice1",
                           "choice2": "choice2",
                           "question": "question"},
            label_classes=["choice1", "choice2"],
            label_column="labels",
            data_url=_SOURCES["ws"],
            data_dir="hulu/ws",
            url=get_repo_url(_SOURCES["ws"])
        ),
        HuluConfig(
            name="sst2",
            description=textwrap.dedent(""""""),
            text_features={"sentence": "sentence"},
            label_classes=["negative", "neutral", "positive"],
            label_column="label",
            data_url=_SOURCES["sst2"],
            data_dir="hulu/sst2",
            url=get_repo_url(_SOURCES["sst2"])
        )
    ]

    def _info(self):
        if self.config.name == "rc":
            features = {
                "query": Value("string"),
                "lead": Value("string"),
                "passage": Sequence(Value("string"))
            }
        else:
            features = {text_feature: Value("string") for text_feature in self.config.text_features.keys()}
        if self.config.label_classes:
            if self.config.name == "rc":
                features["passage_id"] = Value("int32")
                features["start_positions"] = Value("int32")
                features["end_positions"] = Value("int32")
                features["labels"] = Value("string")
            else:
                features["labels"] = ClassLabel(names=self.config.label_classes)
        else:
            features["labels"] = Value("float32")
        features["idx"] = Value("int32")
        return DatasetInfo(
            description=self.config.description,
            features=Features(features),
            homepage=self.config.url,
            citation=textwrap.dedent(_CITATION),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        if self.config.name == "rc":
            path = dl_manager.download(self.config.data_url)
        else:
            path = dl_manager.download_and_extract(self.config.data_url)
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

    def _generate_examples(self, data_file, split_key, **kwargs):
        name = self.config.name
        if name == "rc":
            base_path = data_file[split_key]
        else:
            zip_name = os.listdir(data_file)[0]
            base_path = os.path.join(data_file, zip_name)
        if name == "cola":
            for data in self._cola(base_path, name, split_key):
                yield data
        elif name == "copa":
            for data in self._copa(base_path, name, split_key):
                yield data
        elif name == "wnli":
            for data in self._wnli(base_path, name, split_key):
                yield data
        elif name == "sst2":
            for data in self._sst(base_path, name, split_key):
                yield data
        elif name == "rc":
            for data in self._rc(base_path, name, split_key):
                yield data
        elif name == "ws":
            for data in self._ws(base_path, name, split_key):
                yield data

    @staticmethod
    def _ws(base_path, name, split_key):
        with open(os.path.join(base_path, _PATHS[name][split_key]), mode="r", encoding="utf8") as f:
            content = json.load(f)
            train, test = train_test_split(content, train_size=0.8, random_state=0)
            train, validation = train_test_split(train, train_size=int(len(content) * 0.7), random_state=0)
            if split_key == "train":
                content = train
            elif split_key == "validation":
                content = validation
            elif split_key == "test":
                content = test

        for i, row in enumerate(content):
            label = 0 if row['Answer1'] == row['CorrectAnswer'] else 1
            yield i, {'idx': int(row['ID']), 'question': row['Question'], 'sentence': row['Sent'],
                      'choice1': row['Answer1'], 'choice2': row['Answer2'], 'labels': label}

    @staticmethod
    def _rc(base_path, name, split_key):
        with open(base_path, mode="r", encoding="utf8") as f:
            content = json.load(f)

        for i, row in enumerate(content):
            label = row['MASK'] if 'MASK' in row else -1
            lead = row['lead'][0]
            passage = row['passage']
            query = row['query']
            passage_id, start, end = -1, -1, -1
            if split_key != "test":
                for j, p in enumerate(passage):
                    start = p.find(label)
                    if start != -1:
                        passage_id = j
                        end = start + len(label)
                        break
                else:
                    continue
            yield i, {'idx': int(row['id']), 'lead': lead, 'passage': passage, 'query': query, 'labels': label,
                      'passage_id': passage_id, "start_positions": start, "end_positions": end}

    @staticmethod
    def _sst(base_path, name, split_key):
        with open(os.path.join(base_path, _PATHS[name][split_key]), mode="r", encoding="utf8") as f:
            content = json.load(f)

        for i, row in enumerate(content):
            label = row['Label'] if 'Label' in row else -1
            if label == "negative":
                label = 0
            elif label == "neutral":
                label = 1
            else:
                label = 2
            yield i, {'idx': int(row['Sent_id'].split("_")[-1]), 'sentence': row['Sent'], 'labels': label}

    @staticmethod
    def _wnli(base_path, name, split_key):
        with open(os.path.join(base_path, _PATHS[name][split_key]), mode="r", encoding="utf-8-sig") as f:
            content = json.load(f)
            if split_key == "validation" or split_key == "test":
                content = content["data"]

        for i, row in enumerate(content):
            label = int(row['label']) if 'label' in row else -1
            yield i, {'idx': int(row['id']), 'sentence1': row['sentence1'],
                      'sentence2': row['sentence2'], 'labels': label}

    @staticmethod
    def _copa(base_path, name, split_key):
        with open(os.path.join(base_path, _PATHS[name][split_key]), mode="r", encoding="utf8") as f:
            content = json.load(f)

        for i, row in enumerate(content):
            label = int(row['label'])-1 if 'label' in row else -1
            yield i, {'idx': int(row['idx']), 'question': row['question'], 'premise': row['premise'],
                      'choice1': row['choice1'], 'choice2': row['choice2'], 'labels': label}

    @staticmethod
    def _cola(base_path, name, split_key):
        with open(os.path.join(base_path, _PATHS[name][split_key]), mode="r", encoding="utf8") as f:
            content = json.load(f)['data']

        for i, row in enumerate(content):
            label = int(row['Label']) if 'Label' in row else -1
            yield i, {'idx': int(row['Sent_id'].split("_")[-1]), 'sentence': row['Sent'], 'labels': label}

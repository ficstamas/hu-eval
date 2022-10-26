from datasets import GeneratorBasedBuilder, BuilderConfig, Version, DatasetInfo, Features, Value, \
    Sequence, ClassLabel, DownloadManager, SplitGenerator, Split
import os
import textwrap


_CITATION = """"""
_SOURCE = "https://github.com/nytud/NYTK-NerKor/archive/36a9aefd37e1a77fb8671375def0a5ad343d5dc3.zip"
_PATHS = {
    "train": "data/train-devel-test/train/",
    "validation": "data/train-devel-test/devel/",
    "test": "data/train-devel-test/test/",
}

_SUBS = ["fiction", "legal", "news", "web", "wikipedia"]


_NER = Sequence(
    ClassLabel(
        names=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ]
    )
)

_UPOS = Sequence(
    ClassLabel(
        names=[
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X"
        ]
    )
)


def get_repo_url(x):
    if x.startswith("https://huggingface"):
        return "/".join(x.split("/")[:6])
    return "/".join(x.split("/")[:5])


class HuluConfig(BuilderConfig):
    """BuilderConfig for Hulu."""

    def __init__(
            self,
            text_features,
            data_url,
            data_dir,
            url,
            **kwargs,
    ):
        super(HuluConfig, self).__init__(version=Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.data_url = data_url
        self.data_dir = data_dir
        self.url = url


class NerKor(GeneratorBasedBuilder):
    """NerKor datasets."""
    # FORM LEMMA UPOS XPOS FEATS CONLL:NER
    BUILDER_CONFIGS = [
        HuluConfig(
            name=name,
            description=textwrap.dedent(""""""),
            text_features=["form", "lemma", "upos", "xpos", "feats", "ner"],
            data_url=_SOURCE,
            data_dir=f"nytud-nerkor/",
            url=get_repo_url(_SOURCE)
        ) for name in _SUBS
    ]

    def _info(self):
        features = {
            "idx": Value("int32"),
            "text": Value("string"),
            "tokens": Sequence(Value("string")),
            "lemmas": Sequence(Value("string")),
            "upos": _UPOS,
            "xpos": Sequence(Value("string")),
            "feats": Sequence(Value("string")),
            "ner": _NER,
            "file_name": Value("string"),
            "sentence_id": Value("int32"),
            "morp_tagged": Value("bool_"),
        }
        return DatasetInfo(
            description=self.config.description,
            features=Features(features),
            homepage=self.config.url,
            citation=textwrap.dedent(_CITATION),
        )

    def _split_generators(self, dl_manager: DownloadManager):
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
        base_path = os.path.join(data_file, os.listdir(data_file)[0], _PATHS[split_key], self.config.name)
        train_pointers = [os.path.join(base_path, x) for x in os.listdir(base_path)]
        n = 0
        for annotation in train_pointers:
            for file in os.listdir(annotation):
                pointer_file_path = os.path.join(annotation, file)
                with open(pointer_file_path, mode="r") as f:
                    data_file_path = f.readlines()[0]
                for data in self._process_files(os.path.join(annotation, data_file_path), n):
                    yield data
                    n += 1

    @staticmethod
    def _process_files(data_file_path: str, n: int):
        with open(data_file_path, mode="r", encoding="utf8") as f:
            content = f.readlines()

        tokens, lemmas, upos, xpos, feats, ner = [], [], [], [], [], []

        sentence_id = 0
        for i, row in enumerate(content[1:]):
            if row == '\n':
                p = data_file_path.split("/")[-3:]
                yield n, {
                    "idx": n,
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": feats,
                    "ner": ner,
                    "file_name": "/".join(p),
                    "sentence_id": sentence_id,
                    "morp_tagged": p[1] == 'morph'
                }
                tokens, lemmas, upos, xpos, feats, ner = [], [], [], [], [], []
                sentence_id += 1
                n += 1
                continue

            data = row.split("\t")
            _upos = data[2] if data[2] != "_" else -1
            _xpos = data[3] if data[3] != "_" else -1
            _feats = data[4] if data[4] != "_" else -1
            _ner = data[5] if data[5] != "_" else -1

            tokens.append(data[0])
            lemmas.append(data[1])
            upos.append(_upos)
            xpos.append(_xpos)
            feats.append(_feats)
            ner.append(_ner)

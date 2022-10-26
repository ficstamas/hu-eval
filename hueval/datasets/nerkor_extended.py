from datasets import GeneratorBasedBuilder, BuilderConfig, Version, DatasetInfo, Features, Value, \
    Sequence, ClassLabel, DownloadManager, SplitGenerator, Split
import os
import textwrap
from .nerkor import NerKorConfig, get_repo_url


_CITATION = """"""
_SOURCE = "https://github.com/novakat/NYTK-NerKor-Cars-OntoNotesPP/archive/eb94fc3c22ed27589593716e150d73e060e2333d.zip"
_SUBS = ["fiction", "legal", "news", "web", "wikipedia"]


# Missing tags from readme
_ONPP_TAGS = ["PER", "FAC", "ORG", "GPE", "LOC", "PROD", "EVENT", "WORK_OF_ART", "LAW", "NORP", "LANGUAGE", "DATE",
              "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "AWARD", "CAR", "MEDIA", "SMEDIA",
              "ORG-GPE", "PROJ", "MISC", "MISC-ORG", "MISC-PER", "MISC-LOC", "DUR", "ID", "AGE"]
_ONPP_NER = ClassLabel(
    names=["O"] + [f"{pre}-{tag}" for tag in _ONPP_TAGS for pre in ['B', 'I']]
)


class NerKorExtended(GeneratorBasedBuilder):
    """NerKor 1.41e datasets."""
    # FORM ONPP:NER
    BUILDER_CONFIGS = [
        NerKorConfig(
            name=name,
            description=textwrap.dedent(""""""),
            text_features=["form", "ner"],
            data_url=_SOURCE,
            data_dir=f"nerkor_1_41e/",
            url=get_repo_url(_SOURCE)
        ) for name in _SUBS
    ]

    def _info(self):
        features = {
            "idx": Value("int32"),
            "text": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner": Sequence(_ONPP_NER),
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
        base_path = os.path.join(data_file, os.listdir(data_file)[0], "data")
        n = 0
        split = split_key if split_key != "validation" else "devel"
        files = [x for x in os.listdir(base_path) if split in x and self.config.name in x]
        for file in files:
            for data in self._process_files(os.path.join(base_path, file), n):
                yield data
                n += 1

    @staticmethod
    def _process_files(data_file_path: str, n: int):
        with open(data_file_path, mode="r", encoding="utf8") as f:
            content = f.readlines()

        tokens, ner = [], []

        sentence_id = 0
        for i, row in enumerate(content[1:]):
            if row == '\n':
                p = data_file_path.split("/")[-1]
                yield n, {
                    "idx": n,
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "ner": ner,
                    "file_name": p,
                    "sentence_id": sentence_id,
                    "morp_tagged": not ('no-morph' in p)
                }
                tokens, ner = [], []
                sentence_id += 1
                n += 1
                continue

            data = row.split("\t")

            tokens.append(data[0])
            ner.append(data[1].rstrip('\n'))

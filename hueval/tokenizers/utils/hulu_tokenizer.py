task_to_keys = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
    "opinhubank": ("entity", "sentence")
}


class SequenceTokenizer:
    def __init__(self, tokenizer, task: str, truncation: bool = True,
                 padding: str = "max_length", max_length: int = 512):
        self.tokenizer = tokenizer
        self.task = task
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

    def preprocess_function(self, examples):
        columns = task_to_keys[self.task]
        text_1 = examples[columns[0]]
        text_2 = None if columns[1] is None else examples[columns[1]]
        return self.tokenizer(
            text_1, text_2, truncation=self.truncation, padding=self.padding, max_length=self.max_length
        )


mcqa_task_to_keys = {
    "ws": ("sentence", "question", "choice1", "choice2"),
    "copa": ("premise", "question", "choice1", "choice2")
}


class MultipleChoiceTokenizer:
    def __init__(self, tokenizer, task: str, truncation: bool = True,
                 padding: str = "max_length", max_length: int = 512):
        self.tokenizer = tokenizer
        self.task = task
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

    def preprocess_function(self, examples):
        columns = mcqa_task_to_keys[self.task]
        num_choices = len(columns[2:])
        first_sentences = [[f"{x}" if self.task == "copa" else f"{x} {y}"] * len(columns[2:]) for x, y in zip(examples[columns[0]], examples[columns[1]])]

        second_sentences = [
            [f"{examples[choice][i]}" for choice in columns[2:]] for i in range(len(examples[columns[0]]))
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        tokenized = self.tokenizer(
            first_sentences, second_sentences, truncation=self.truncation, padding=self.padding, max_length=self.max_length
        )
        return {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized.items()}

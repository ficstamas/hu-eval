class AlignLabels:
    def __init__(self, tokenizer, label_name: str, label_all_tokens: bool = False, truncation: bool = True,
                 is_split_into_words: bool = True, padding: str = "max_length", max_length: int = 512):
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.label_all_tokens = label_all_tokens
        self.truncation = truncation
        self.is_split_into_words = is_split_into_words
        self.padding = padding
        self.max_length = max_length

    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=self.truncation, is_split_into_words=self.is_split_into_words,
            padding=self.padding, max_length=self.max_length
        )

        labels = []
        for i, label in enumerate(examples[self.label_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

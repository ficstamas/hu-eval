from hueval.models.hubert import _TYPES, download_and_extract
from transformers import BertTokenizer
import os


class HuBertUncasedTokenizer(BertTokenizer):
    text_arguments = ['text', 'text_pair', 'text_target', 'text_pair_target']

    def __call__(self, *args, **kwargs):
        new_args = []
        for i in range(min(len(args), 4)):
            if type(args[i]) is str:
                new_args.append(args[i].lower())
            elif type(args[i]) is list:
                if type(args[i][0]) is list:
                    new_args.append([[x.lower() for x in y] for y in args[i]])
                else:
                    new_args.append([x.lower() for x in args[i]])

        for argument_name in self.text_arguments[i+1:]:
            if argument_name in kwargs:
                if type(kwargs[argument_name]) is str:
                    kwargs[argument_name] = kwargs[argument_name].lower()
                elif type(kwargs[argument_name]) is list:
                    if type(kwargs[argument_name][0]) is list:
                        kwargs[argument_name] = [[x.lower() for x in y] for y in kwargs[argument_name]]
                    else:
                        kwargs[argument_name] = [x.lower() for x in kwargs[argument_name]]
        return super(HuBertUncasedTokenizer, self).__call__(*new_args, **kwargs)


# "model_type": "bert"

def load_tokenizer(model_type: _TYPES, return_wrapped: bool = True) -> BertTokenizer:
    path = download_and_extract(model_type)
    prefix = "hubert_wiki" if model_type == "cased" else "hubert_wiki_lower"
    path = os.path.join(path, prefix)
    if model_type == "uncased":
        if return_wrapped:
            tokenizer = HuBertUncasedTokenizer.from_pretrained(path, do_lower_case=False)
        else:
            tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(path)

    return tokenizer

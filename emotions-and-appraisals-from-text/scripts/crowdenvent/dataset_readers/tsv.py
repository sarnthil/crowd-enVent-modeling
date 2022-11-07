from typing import Dict, Iterable, List, Optional

import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, TensorField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
import csv

# TODO: handle reg vs cls transformation of the appraisal values
@DatasetReader.register("tsv")
class TsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        # x_column: str = "generated_text",   # previous classifier saw more emotion words...
        x_column: str = "hidden_emo_text",
        x_tra_columns: Optional[List[str]] = None,
        y_column: str = "emotion",
        binning: bool = False,
        scaling: bool = False,
        x_tra_in_text: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.x_column, self.y_column, self.x_tra_columns = x_column, y_column, x_tra_columns
        self.x_tra_in_text = x_tra_in_text
        self.binning = binning
        self.scaling = scaling

    def text_to_instance(self, text: str, label: str = None, x_tra_columns: Optional[List[str]] = None) -> Instance:
        if self.x_tra_in_text:
            text = " ".join(k for k, v in x_tra_columns if v >= 4) + " " + text
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}
        if x_tra_columns and not self.x_tra_in_text:
            fields["extra"] = TensorField(np.array(list(x_tra_columns.values())))
        if label:
            if self.binning:
                label = "high" if int(label) >= 4 else "low"
            if self.scaling:
                label = str((int(label) - 1) / 4)
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as f:
            lines = csv.DictReader(f, delimiter="\t")
            for line in lines:
                x = line[self.x_column]
                y = line[self.y_column]
                z = {col: int(line[col]) for col in (self.x_tra_columns or [])}
                yield self.text_to_instance(x, y, z)

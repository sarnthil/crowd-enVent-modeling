import os
import random
import json
from typing import List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import single_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_json, write_json


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.FloatTensor
    tokens: list


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label=self.label,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label: float

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label=self.label,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label: float
    tokens: list


class XenventSuddennessTask(Task):
    TASK_TYPE = TaskTypes.REGRESSION

    Batch = Batch
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow

    def get_train_examples(self):
        return self._create_examples(path=self.path_dict["data"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.path_dict["data"], set_type="dev")

    def get_test_examples(self):
        return self._create_examples(path=self.path_dict["data"], set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        df = pd.read_csv(path, index_col=0, names=["split", "suddenness", "text", ...])
        for i, row in df.iterrows():
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=row.text,
                    label=row.suddenness if set_type != "test" else 0,
                )
            )
        return examples

        # raw_examples = []
        # with open(path) as f:
        #     for line in f:
        #         data = json.loads(line)
        #         if data["dataset"] != "gne":
        #             continue
        #         if data["split"] != set_type:
        #             continue
        #         raw_examples.append(data)
        # random.shuffle(raw_examples)
        # examples = []
        # for i, data in enumerate(raw_examples):
        #     examples.append(
        #         Example(
        #             guid="%s-%s" % (set_type, i),
        #             text=data["text"],
        #             label=float(line["suddenness"]) if set_type != "test" else 0,
        #         )
        #     )
        # return examples

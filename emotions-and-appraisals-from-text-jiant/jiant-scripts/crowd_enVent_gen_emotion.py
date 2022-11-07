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
    label_id: torch.LongTensor
    tokens: list


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            # text=self.text.split(" "),
            text=tokenizer.tokenize(self.text),
            label_id=enventClassificationTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
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
    label_id: int
    tokens: list


class enventClassificationTask(Task):
    # emotion classification
    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [
        "joy",
        "sadness",
        "surprise",
        "anger",
        "fear",
        "disgust",
        "relief",
        "guilt",
        "shame",
        "trust",
        "pride",
        "no-emotion",
        "boredom",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
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
        df = pd.read_csv(path, index_col=0, names=["split", "label", "text"])
        for i, row in df.iterrows():
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=row.text,
                    label=row.label if set_type != "test" else cls.LABELS[-1],
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
        #             label=data["emotions"][0],
        #         )
        #     )
        # return examples

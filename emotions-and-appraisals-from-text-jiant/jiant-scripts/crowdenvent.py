import os
import re
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


ACTIVE_LABEL_MAP = None


@dataclass
class ClsBatch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


@dataclass
class RegBatch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.FloatTensor
    tokens: list

    @property
    def label(self):
        return self.label_id

@dataclass
class ClsExample(BaseExample):
    guid: str
    text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedClsExample(
            guid=self.guid,
            # text=self.text.split(" "),
            text=tokenizer.tokenize(self.text),
            label_id=ACTIVE_LABEL_MAP[self.label],
        )


@dataclass
class TokenizedClsExample(BaseTokenizedExample):
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
            data_row_class=ClsDataRow,
        )


@dataclass
class RegExample(BaseExample):
    guid: str
    text: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedRegExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=self.label,
        )

    @property
    def label(self):
        return self.label_id

@dataclass
class TokenizedRegExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: float

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=RegDataRow,
        )

    @property
    def label(self):
        return self.label_id


@dataclass
class ClsDataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class RegDataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: float
    tokens: list

    @property
    def label(self):
        return self.label_id


class CrowdEnventEmotionClassification(Task):
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
        "boredom",
        "no-emotion",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
    Batch = ClsBatch
    Example = ClsExample
    TokenizedExample = TokenizedClsExample
    DataRow = ClsDataRow

    def get_train_examples(self):
        return self._create_examples(path=self.path_dict["train"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.path_dict["val"], set_type="dev")

    def get_test_examples(self):
        return self._create_examples(path=self.path_dict["test"], set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        global ACTIVE_LABEL_MAP
        ACTIVE_LABEL_MAP = cls.LABEL_TO_ID
        df = pd.read_csv(path, sep="\t")
        examples = []
        for i, row in df.iterrows():
            examples.append(
                ClsExample(
                    guid="%s-%s" % (set_type, i),
                    text=row.generated_text,
                    label=row.emotion if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples


def binarize(label):
    return "high" if int(label) >= 4 else "low"


class AppraisalClassificationBase(Task):
    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [
        "high",
        "low",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
    Batch = ClsBatch
    Example = ClsExample
    TokenizedExample = TokenizedClsExample
    DataRow = ClsDataRow

    def get_train_examples(self):
        return self._create_examples(path=self.path_dict["train"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.path_dict["val"], set_type="dev")

    def get_test_examples(self):
        return self._create_examples(path=self.path_dict["test"], set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        global ACTIVE_LABEL_MAP
        ACTIVE_LABEL_MAP = cls.LABEL_TO_ID
        df = pd.read_csv(path, sep="\t")
        examples = []
        for i, row in df.iterrows():
            examples.append(
                ClsExample(
                    guid="%s-%s" % (set_type, i),
                    text=row.generated_text,
                    label=binarize(getattr(row, cls.appraisal))
                    if set_type != "test"
                    else cls.LABELS[-1],
                )
            )
        return examples


class AppraisalRegressionBase(Task):
    TASK_TYPE = TaskTypes.REGRESSION

    Batch = RegBatch
    Example = RegExample
    TokenizedExample = TokenizedRegExample
    DataRow = RegDataRow

    def get_train_examples(self):
        return self._create_examples(path=self.path_dict["train"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.path_dict["val"], set_type="dev")

    def get_test_examples(self):
        return self._create_examples(path=self.path_dict["test"], set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        df = pd.read_csv(path, sep="\t")
        for i, row in df.iterrows():
            examples.append(
                RegExample(
                    guid="%s-%s" % (set_type, i),
                    text=row.generated_text,
                    label=getattr(row, cls.appraisal)/5 if set_type != "test" else 0,
                )
            )
        return examples

def pascalify(match):
    return match.group()[-1].upper()

for appraisal in [
    "suddenness",
    "familiarity",
    "predict_event",
    "pleasantness",
    "unpleasantness",
    "goal_relevance",
    "chance_responsblt",
    "self_responsblt",
    "other_responsblt",
    "predict_conseq",
    "goal_support",
    "urgency",
    "self_control",
    "other_control",
    "chance_control",
    "accept_conseq",
    "standards",
    "social_norms",
    "attention",
    "not_consider",
    "effort",
]:
    appraisal_title = re.sub(r"^\w|_\w", pascalify, appraisal)

    for kind, base in (("Classification", AppraisalClassificationBase), ("Regression", AppraisalRegressionBase)):
        class ThisAppraisalTask(base):
            appraisal = appraisal

        class_name = f"CrowdEnvent{appraisal_title}{kind}"
        ThisAppraisalTask.__name__ = class_name
        locals()[class_name] = ThisAppraisalTask

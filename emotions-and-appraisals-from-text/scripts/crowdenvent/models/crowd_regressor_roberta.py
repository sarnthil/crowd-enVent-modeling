from typing import Dict, Optional
from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import PearsonCorrelation, SpearmanCorrelation
from typing import Dict
from overrides import overrides
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError, PearsonCorrelation

import torch
from torch.nn import MSELoss
from transformers import AutoConfig, AutoModel, AutoTokenizer

import logging


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("crowd_regressor_roberta")
class CrowdRegressorRoberta(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "roberta-base",
        hidden_size: int = 1,
        train_base: bool = True,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)

        if not train_base:
            for _, param in self.transformer.named_parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )
        self.regression_layer = [
            torch.nn.Linear(self.embedding_dim, hidden_size),
        ]
        if hidden_size > 1:
            self.regression_layer.extend([
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_size, 1),
                ])
        self.regression_layer = torch.nn.Sequential(*self.regression_layer)
        self.loss = MSELoss()
        self._mae_metric = MeanAbsoluteError()
        self._pearson_metric =  PearsonCorrelation()
        self._spearman_metric =  SpearmanCorrelation()

    @property
    def embedding_dim(self):
        return self.transformer.config.hidden_size

    def forward(
        self,
        text: TextFieldTensors,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        input_ids = util.get_token_ids_from_text_field_tensors(text)
        attention_mask = util.get_text_field_mask(text)

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # (batch_size, lf_hidden_size)
        regression_input = output['pooler_output']
        # (batch_size, 1)
        prediction = self.regression_layer(regression_input)
        prediction = torch.sigmoid(prediction)
        # _, prediction = torch.max(prediction,1)

        output_dict = {
            "predicted_appraisal": prediction,
        }


        if label is not None:
            label = label.float()
            loss = self.loss(prediction, label)
            output_dict["loss"] = loss
            output_dict["label"] = label

            self._mae_metric(prediction, label)
            # self._pearson_metric(prediction, label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        result = self._mae_metric.get_metric(reset)
        result["pearson"] =  self._pearson_metric.get_metric(reset)
        result["spearman"] =  self._spearman_metric.get_metric(reset)
        return result




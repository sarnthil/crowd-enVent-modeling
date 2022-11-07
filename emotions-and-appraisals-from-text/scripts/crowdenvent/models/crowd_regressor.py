from typing import Dict, Optional
from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import PearsonCorrelation, SpearmanCorrelation

@Model.register("crowd_regressor")
class CrowdRegressor(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 dropout: float = None,
                 scale: float = 1,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        self._num_labels = 1  # because we're running a regression task
        self._scale = scale

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._metric = PearsonCorrelation()
        self._metric2 = SpearmanCorrelation()
        self._loss = torch.nn.MSELoss()
        initializer(self)

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        embedded_text = self._text_field_embedder(text)
        mask = get_text_field_mask(text).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        output_dict = {"logits": logits}

        if label is not None:  # convert the label into a float number and update the metric
            label_to_str = lambda l: self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(l)
            label_tensor = torch.tensor([float(label_to_str(int(label[i]))) for i in range(label.shape[0])], device=logits.device)
            loss = self._loss(logits.view(-1), label_tensor)
            output_dict["loss"] = loss
            self._metric(logits, label_tensor)
            self._metric2(logits, label_tensor)

        return output_dict

    # def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # update this part to generate a float number result as similarity score
        predictions = output_dict["logits"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = "{:.1f}".format(prediction.long())
            label_str = (self.vocab.get_index_to_token_vocabulary(self._label_namespace)
                         .get(label_idx, str(label_idx)))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'PearsonCorrelation': self._metric.get_metric(reset), 'SpearmanCorrelation': self._metric2.get_metric(reset)}
        return metrics


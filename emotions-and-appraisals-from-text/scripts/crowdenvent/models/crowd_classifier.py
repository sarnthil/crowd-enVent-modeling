from typing import Dict

import numpy
import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure

@Model.register("crowd_classifier")
class CrowdClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, x_tra_columns = None
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        label_namespace: str = "labels"
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim() + (len(x_tra_columns) if x_tra_columns else 0),
            num_labels,
        )
        self.accuracy = CategoricalAccuracy()
        self.f1 = FBetaMeasure()
        self.label_tokens = vocab.get_index_to_token_vocabulary(label_namespace)
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None, extra: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        if extra is not None:
            encoded_text = torch.cat([encoded_text, extra], -1)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs, "logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1(logits, label)
            # output["loss"] = torch.nn.functional.cross_entropy(logits, label)
            output["loss"] = self._loss(logits, label.long().view(-1))
        return output

    # def make_output_human_readable(self, output_dict):
    #     preds = output_dict["probs"]
    #     if len(preds.shape) == 1:
    #         output_dict["probs"] = preds.unsqueeze(0)
    #         output_dict["logits"] = output_dict["logits"].unsqueeze(0)

    #     classes = []
    #     for prediction in output_dict["probs"]:
    #         label_idx = prediction.argmax(dim=-1).item()
    #         output_dict["loss"] = self._loss(output_dict["logits"], torch.LongTensor([label_idx]))
    #         label_str = str(label_idx)
    #         classes.append(label_str)
    #     output_dict["label"] = classes
    #     return output_dict

    def make_output_human_readable(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a `"label"` key to the dictionary with the result.
        """
        predictions = output["probs"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output["label"] = labels
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        all_metrics.update({"accuracy": self.accuracy.get_metric(reset)})
        # all_metrics.update({"f1": self.f1.get_metric(reset=reset)["fscore"]})
        for metric_name, metrics_per_class in self.f1.get_metric(reset).items():
            # import ipdb; ipdb.set_trace()
            for class_index, value in enumerate(metrics_per_class):
                all_metrics[f"{self.label_tokens[class_index]}-{metric_name}"] = value
        return all_metrics

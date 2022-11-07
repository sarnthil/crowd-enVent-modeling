from typing import List
from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register("crowd_classifier")
class CrowdClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        instance = self._json_to_instance(inputs)
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    @overrides
    def predictions_to_labeled_instances(self, instance, outputs):
        return [instance]

#!/pfs/data5/home/st/st_us-052440/st_ac131284/venvs/allennlp/bin/python
import json
from pathlib import Path
from time import monotonic

import click
import numpy as np
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.common.params import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from crowdenvent.dataset_readers.tsv import TsvReader
from crowdenvent.predictors.crowd_classifier_predictor import CrowdClassifierPredictor
from lime.lime_text import LimeTextExplainer

EMOTIONS = [
    "anger",
    "boredom",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "no-emotion",
    "pride",
    "relief",
    "sadness",
    "shame",
    "surprise",
    "trust",
]

@click.command()
@click.option("--corpus", "-c")
@click.option("--skip", "-s", default=0)
@click.argument("model_path", type=click.Path())
def cli(corpus, model_path, skip):
    model_path = Path(model_path)
    with (model_path / "config.json").open() as f:
        config = Params(json.load(f))
    model = Model.load(config, str(model_path))
    tokenizer = Tokenizer.from_params(config["dataset_reader"]["tokenizer"])

    x_tra_columns = config["dataset_reader"].get("x_tra_columns"),
    kwargs = {"x_tra_columns": x_tra_columns} if x_tra_columns else {}
    reader = TsvReader(
        tokenizer=tokenizer,
        token_indexers={"bert": TokenIndexer.from_params(config["dataset_reader"]["token_indexers"]["bert"])},
        max_tokens=config["dataset_reader"]["max_tokens"],
        y_column=config["dataset_reader"]["y_column"],
        # binning=config["dataset_reader"]["binning"],
        # **kwargs
    )
    predictor = CrowdClassifierPredictor(
        model=model,
        dataset_reader=reader,
    )
    # predictor = CrowdClassifierPredictor.from_path(model)
    # reader = TsvReader()
    explainer = LimeTextExplainer(class_names=EMOTIONS)

    def predict_probabilities(sentences):
        return np.array([
            # predictor.predict_instance(reader.text_to_instance(sentence))['probs']
            predictor.predict(sentence)['probs']
            for sentence in sentences
        ])

    with (model_path / "lime-explanation.json").open("a") as output:
        for instance in reader._read(corpus):
            if skip > 0:
                skip -= 1
                continue
            start = monotonic()
            instance_data = instance.human_readable_dict()
            sentence = " ".join(instance_data["text"])
            explanation = explainer.explain_instance(sentence, predict_probabilities, num_features=len(instance_data["text"]), num_samples=100, labels=list(range(13)))
            print(
                json.dumps(
                    {
                        "sentence": sentence,
                        "gold_label": instance_data["label"],
                        "explanations": {
                            emotion: explanation.as_list(label=label)
                            for label, emotion in enumerate(EMOTIONS)
                        },
                    }
                ),
                flush=True,
                file=output,
            )
            print(monotonic() - start)

if __name__ == '__main__':
    cli()

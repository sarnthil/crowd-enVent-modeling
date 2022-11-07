import json
import csv
import sys
import click
from pathlib import Path

from allennlp.interpret.saliency_interpreters import SimpleGradient, IntegratedGradient
from allennlp.models import Model
from allennlp.common.params import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

import crowdenvent
from crowdenvent.dataset_readers import TsvReader
from crowdenvent.predictors import CrowdClassifierPredictor

# Saliency maps interact with the model through the input tokens' embeddings. It
# essentially will return a normalized score (calculated from each embedding's
# gradient) that represents how sensitive the model is towards the change in the
# tokens' embeddings. The higher the score, the more sensitive the model is. By
# "normalized" we mean, after aggregation, the scores are weighted to make sure
# the scores sum up to 1.

# Vanilla Gradient (SimpleGradient), Integrated Gradient (IntegratedGradient),
# Smooth Gradient (SmoothGradient).

@click.command()
@click.argument("model_path", type=click.Path())
@click.option("--data", type=click.File())
@click.option("--interpret/--no-interpret", default=True, is_flag=True)
def cli(data, model_path, interpret):
    model_path = Path(model_path)
    # inputs = {"sentence": "An art piece I produced was auctioned off for charity and raised a good sum for the cause."}
    # add a test running example

    with (model_path / "config.json").open() as f:
        config = Params(json.load(f))
    model = Model.load(config, str(model_path))
    tokenizer = Tokenizer.from_params(config["dataset_reader"]["tokenizer"])

    x_tra_columns = config["dataset_reader"].get("x_tra_columns"),
    kwargs = {"x_tra_columns": x_tra_columns} if x_tra_columns else {}
    predictor = CrowdClassifierPredictor(
        model=model,
        dataset_reader=TsvReader(
            tokenizer=tokenizer,
            token_indexers={"bert": TokenIndexer.from_params(config["dataset_reader"]["token_indexers"]["bert"])},
            max_tokens=config["dataset_reader"]["max_tokens"],
            y_column=config["dataset_reader"]["y_column"],
            binning=config["dataset_reader"]["binning"],
            **kwargs
        )
    )
    if interpret:
        interpreter_emotion = IntegratedGradient(predictor)

    reader = csv.DictReader(data, delimiter="\t")
    rows = (row for row in reader)

    # we callect all of them
    with (model_path / "interpret-integrated-hidden-emo.txt").open("w") as f:
        for row in rows:
            inputs = {"sentence": row["hidden_emo_text"]}
            if x_tra_columns:
                inputs["x_tra_columns"] = {col: int(row[col]) for col in x_tra_columns}
            prediction = model.label_tokens[int(predictor.predict_json(inputs)["label"])]
            if interpret:
                interpretation_emotion = interpreter_emotion.saliency_interpret_from_json(inputs)
                print(prediction, json.dumps([[tok.text, val] for tok, val in zip(tokenizer.tokenize(sentence), interpretation_emotion["instance_1"]["grad_input_1"])]), sep="\t", file=f)
            else:
                print(prediction, file=f)

if __name__ == "__main__":
    cli()

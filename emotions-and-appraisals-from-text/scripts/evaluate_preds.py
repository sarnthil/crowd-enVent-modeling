import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, StatisticsError, stdev

import click
from sklearn.metrics import classification_report
from scipy.stats import pearsonr

EMOTIONS = [
    "anger", "boredom", "disgust", "fear", "guilt", "joy", "no-emotion",
    "pride", "relief", "sadness", "shame", "surprise", "trust",
]


def get_labels(file, format):
    for line in file:
        if format == "interpret":
            yield line.split("\t")[0].strip()
        elif format in ("json", "json-silver"):
            yield json.loads(line)["label"]
        else:
            raise ValueError(f"Unknown format {format}")


def read_preds(file, format):
    for val in get_labels(file, format):
        if val == "high":
            yield 1
        elif val == "low":
            yield 0
        elif val.isdigit():
            yield 1 if int(val) >= 4 else 0
        else:
            yield val

def read_gold(file, column):
    for row in csv.DictReader(file, delimiter="\t"):
        val = row[column]
        if val == "high":
            yield 1
        elif val == "low":
            yield 0
        elif val.isdigit():
            yield 1 if int(val) >= 4 else 0
        else:
            yield val


def column_from_model_name(model_name):
    # classification-self_responsblt-model-roberta-23/
    return model_name.split("-")[1]


def column_and_files(format):
    for model in Path("../workdata").glob("*-*"):
        if not model.is_dir():
            continue
        # FIXME:
        if "emotion" not in model.name:
            continue

        seed = int(model.name.split("-")[-1])

        column = column_from_model_name(model.name)
        if format == "interpret":
            interpret_path = model / "interpret-integrated-hidden-emo.txt"
        elif format == "json":
            interpret_path = model / "predictions.json"
        elif format == "json-silver":
            interpret_path = model / "predictions_silver.json"
        else:
            raise ValueError(f"Unknown format {format}")
        if interpret_path.exists():
            if "regression" in str(interpret_path):
                continue  # FIXME
            yield column, Path("../sources/crowd-enVent_validation_deduplicated.tsv"), interpret_path, seed  # FIXME: doesn't belong in sources!!

def get_mean(emotion, result, metric):
    # result[23]
    # {'anger': {'precision': 0.2197452229299363, 'recall': 0.69, 'f1-score': 0.3333333333333333, 'suppor
    return mean(seed_data[emotion][metric] for seed_data in result.values()) * 100
    # avg = round(mean(seed_data[emotion][metric] for seed_data in result.values()) * 100)
    # return f"${avg}$"

def get_stdev(emotion, result, metric):
    # result[23]
    # {'anger': {'precision': 0.2197452229299363, 'recall': 0.69, 'f1-score': 0.3333333333333333, 'suppor
    return stdev(seed_data[emotion][metric] for seed_data in result.values())
    # avg = round(mean(seed_data[emotion][metric] for seed_data in result.values()) * 100)
    # return f"${avg}$"


@click.command()
@click.option("--format", type=click.Choice(["json", "json-silver", "interpret"]), default="json")
@click.option("--model", default="")
def cli(format, model):
    results = defaultdict(dict)
    for column, true_path, pred_path, seed in column_and_files(format):
        if model not in str(pred_path.resolve()):
            continue
        with true_path.open() as true_file, pred_path.open() as pred_file:
            print("Evaluation for", column, "/", pred_path.parent.name)
            y_true = list(read_gold(true_file, column))
            y_pred = list(read_preds(pred_file, format))
            if column == "emotion":
                results[
                    "combined" if "combined" in str(pred_path) else "classification"
                ][seed] = classification_report(y_true, y_pred, output_dict=True)
            else:
                print(pearsonr(y_true, y_pred))
                print()
    for part in ["combined", "classification"]:
        precisions = []
        recalls = []
        fscores = []
        print("==========", part, "============")
        try:
            for emotion in EMOTIONS:
                precision, recall, fscore = (
#                    get_mean(emotion, results[part], "precision"),
#                    get_mean(emotion, results[part], "recall"),
#                    get_mean(emotion, results[part], "f1-score"),
                    get_stdev(emotion, results[part], "precision"),
                    get_stdev(emotion, results[part], "recall"),
                    get_stdev(emotion, results[part], "f1-score"),
                )
                precisions.append(precision)
                recalls.append(recall)
                fscores.append(fscore)

                print(
                    f"${(precision)}$",
                    f"${(recall)}$",
                    f"${(fscore)}$",
                    #f"${round(precision)}$",
                    #f"${round(recall)}$",
                    #f"${round(fscore)}$",
                    sep=" & ",
                )
            print("-------------------------------")
            print(
                f"${round(mean(precisions))}$",
                f"${round(mean(recalls))}$",
                f"${round(mean(fscores))}$",
                sep=" & ",
            )
            print("===============================")
            print("stdevs... PRF")
            print(
                f"${(stdev(precisions))}$",
                f"${(stdev(recalls))}$",
                f"${(stdev(fscores))}$",
                sep=" & ",
            )
            print("===============================")
        except StatisticsError:
            pass

if __name__ == "__main__":
    cli()

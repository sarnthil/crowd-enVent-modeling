# We model how well do the validators predict the emotion and the appraisals
import sys
import math
import json
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score, classification_report
from scipy.stats import spearmanr
from collections import defaultdict
from statistics import mean

random.seed(0)

# TODO fix this
BASE_PATH = Path("/Users/sarnthil/CEAT/crowdsourcing-appraisals/outputs/corpus/")

# choose from "f1" or "spearman"
# F1_OR_SPEARMAN = "f1"

# columns in generation
APPRAISALS = {
    "suddenness": "Suddenness",
    "familiarity": "Familiarity",
    "predict_event": "Predict. Event",
    "pleasantness": "Pleasantness",
    "unpleasantness": "Unpleasantness",
    "goal_relevance": "Goal Relev.",
    "chance_responsblt": "Resp. Chance",
    "self_responsblt": "Resp. Self",
    "other_responsblt": "Resp. Other",
    "predict_conseq": "Predict Conseq.",
    "goal_support": "Goal Support",
    "urgency": "Urgency",
    "self_control": "Control Self",
    "other_control": "Control Other",
    "chance_control": "Control Chance",
    "accept_conseq": "Accept Conseq.",
    "standards": "Standards",
    "social_norms": "Social Norms",
    "attention": "Attention",
    "not_consider": "Not Consider",
    "effort": "Effort",
}

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

def clamp(value, key):
    if key == "emotion" or F1_OR_SPEARMAN == "spearman":
        return int(value) if isinstance(value, str) and value.isdigit() else value
    return 1 if int(value) >= 4 else 0

def filter(row):
    return True
    # return int(row["event_familiarity"]) < 4

def resolve(scores, key):
    """Given a list of scores for a single instance, select the correct one"""
    values = Counter(instance["value"] for instance in scores)
    if len(values) == 1:
        # everyone agrees
        return scores[0]["value"]
    top, second = values.most_common(2)
    if top[1] > second[1]:
        # majority vote
        return top[0]
    # now we give them as many votes as they are confident
    values = Counter(
        instance["value"]
        for instance in scores
        for _ in range(instance["confidence"])
    )
    top, second = values.most_common(2)
    if top[1] > second[1]:
        # majority vote
        return top[0]
    # now we give them as many votes as their intensities as well
    values = Counter(
        instance["value"]
        for instance in scores
        for _ in range(instance["intensity"] + instance["confidence"])
    )
    top, second = values.most_common(2)
    if top[1] > second[1]:
        # majority vote
        return top[0]
    # now we give them as many votes as the event is familiar to the annotator
    values = Counter(
        instance["value"]
        for instance in scores
        for _ in range(instance["intensity"] + instance["confidence"] + instance["event_familiarity"])
    )
    top, second = values.most_common(2)
    if top[1] > second[1]:
        # majority vote
        return top[0]
    else:
        # finally, pick randomly between ties
        return random.choice([key for key, val in values.items() if val == top[1]])

# columns in validators "original_emotion" vs "emotion"


def collect_from_report(appraisal, report):
    for line in report.split("\n"):
        if line.strip().startswith("1"):
            _, p, r, f, _ = (int(float(value) * 100) for value in line.split())
            return {
                appraisal: {
                    "precision": p,
                    "recall": r,
                    "fscore": f,
                }
            }



@click.command()
@click.option("--dump", is_flag=True)
@click.option("--dump-raw", type=click.File("w"))
@click.option("--metric", type=click.Choice(["f1", "spearman"]), default="f1")
def cli(dump, dump_raw, metric):
    global F1_OR_SPEARMAN, KEYS
    F1_OR_SPEARMAN = metric
    KEYS = ["emotion", *APPRAISALS] if F1_OR_SPEARMAN != "spearman" else APPRAISALS

    original_emotions = {}
    with (BASE_PATH / "crowd-enVent_generation.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        generated = defaultdict(dict)
        for row in reader:
            # {"text_id_213213": {"emotion": "sadness", "suddenness": 3, ...}, ...}
            for key in KEYS:
                generated[key][row["text_id"]] = clamp(row[key], key)
    generated = dict(generated)

    with (BASE_PATH / "crowd-enVent_validation.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        judgments = defaultdict(dict)
        for row in reader:
            if not filter(row):
                continue
            text_id = row["text_id"]
            for key in KEYS:
                # "predicted":
                judgments[key].setdefault(text_id, []).append(
                    {
                        "value": clamp(row[key], key),
                        "confidence": int(row["confidence"]),
                        "intensity": int(row["intensity"]),
                        "round_numer": row["round_number"],
                        "event_familiarity": int(row["event_familiarity"]),
                    }
                )
    judgments = dict(judgments)
    text_ids = set()

    for key in judgments:
        for text_id in judgments[key]:
            judgments[key][text_id] = resolve(judgments[key][text_id], key)
            text_ids.add(text_id)

    if dump_raw:
        fields = ["text_id"]
        if "emotion" in KEYS:
            fields.append("original_emotion")
        fields.extend(key.replace("emotion", "predicted_emotion") for key in KEYS)

        writer = csv.DictWriter(dump_raw, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for text_id in sorted(text_ids, key=int):
            row = {
                "text_id": text_id,
                **{
                    key.replace("emotion", "predicted_emotion"): judgments[key][text_id]
                    for key in KEYS
                },
            }
            if "emotion" in KEYS:
                row["original_emotion"] = generated["emotion"][text_id]
            writer.writerow(row)
        sys.exit()

    dump_data = {}
    for key in KEYS:
        y_true = [generated[key][text_id] for text_id in judgments[key]]
        y_pred = list(judgments[key].values())
        if key == "emotion":
            if dump:
                continue
            scores = f1_score(y_true, y_pred, average=None, labels=EMOTIONS)
            rscores = recall_score(y_true, y_pred, average=None, labels=EMOTIONS)
            pscores = precision_score(y_true, y_pred, average=None, labels=EMOTIONS)
            # print("emotion precision scores:", {emotion: score for emotion, score in zip(EMOTIONS, pscores)})
            # print("emotion recall scores:", {emotion: score for emotion, score in zip(EMOTIONS, rscores)})
            # print("emotion f1 scores:", {emotion: score for emotion, score in zip(EMOTIONS, scores)})
            print(classification_report(y_true, y_pred, target_names=EMOTIONS))
            print("emotion micro-precision:", precision_score(y_true, y_pred, average='micro'))
            print("emotion micro-recall:", recall_score(y_true, y_pred, average='micro'))
            print("emotion micro-f1:", f1_score(y_true, y_pred, average='micro'))
        else:
            if dump:
                dump_data.update(collect_from_report(key, classification_report(y_true, y_pred)))
            else:
                print("===============")
                print(key, "&", round(f1_score(y_true, y_pred, average="binary"),2))
                print(classification_report(y_true, y_pred))
                print("===============")
        # if F1_OR_SPEARMAN == "f1":
        #     print(key, "&", round(f1_score(y_true, y_pred, average="macro"),2))
        #     # print(key, "&", round(f1_score(y_true, y_pred, average="micro"),2))
        #     # print(classification_report(y_true, y_pred))
        #     # print(key, "&", round(precision_score(y_true, y_pred, average="macro"),2))
        #     # print(key, "&", round(recall_score(y_true, y_pred, average="micro"),2))
        # elif F1_OR_SPEARMAN == "spearman":
        #     correlation, pvalue = spearmanr(y_true, y_pred)
        #     print(key, "&", f"{correlation:.2f}")
        # else:
        #     raise ValueError(f"Unknown value {F1_OR_SPEARMAN=}")
    if dump:
        print(json.dumps(dump_data))


if __name__ == '__main__':
    cli()

"""Calculate F1-score per emotion for the single case"""

import csv
import json
from collections import Counter, defaultdict
from itertools import zip_longest

import click
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report

# get mapping from id2label cause preds are numbers
emotions = [
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
id2label = dict(enumerate(emotions))

confusion = defaultdict(Counter)

def JSONReader(file):
    for line in file:
        yield json.loads(line)


@click.command()
@click.argument("f_pred", type=click.File("r"))
@click.option("--output", "-o", required=True)
def cli(f_pred, output):
    preds, golds = [], []
    with open("workdata/crowd-enVent_validation_deduplicated.tsv") as f_gold:
        r_gold = csv.DictReader(f_gold, delimiter="\t")
        r_pred = JSONReader(f_pred)
        for gold, pred in zip_longest(r_gold, r_pred, fillvalue="ERROR: unequal size"):
            prediction = pred["label"]
            truth = gold["emotion"]
            preds.append(prediction)
            golds.append(truth)
            confusion[truth][prediction] += 1

    print(classification_report(golds, preds))

    df = pd.DataFrame(
        [[confusion[i][j] for j in emotions] for i in emotions],
        columns=emotions,
        index=emotions,
    )
    plt.rcParams.update({'font.size': 10,})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(golds, preds)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=emotions)
    cm.plot(values_format="d", cmap="Blues", ax=ax)
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #              ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output)

if __name__ == '__main__':
    cli()

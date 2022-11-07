import json
import csv
from pathlib import Path
from itertools import cycle
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import spearmanr, pearsonr

EMOTIONS = dict(enumerate([
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
]))

def map_gold(value):
    # map apparent ints to ints, but leave "real" strings as-is
    if value.isdigit():
        return int(value)
    return value

def map_pred(value, *, column, cls_or_reg):
    if column == "emotion":
        return EMOTIONS[int(value)]
    elif cls_or_reg == "cls":
        return int(value)
    else:
        return float(value)


for run in Path("workdata/runs").iterdir():
    # run-single-accept-conseq-reg etc.
    if not run.name.startswith("run-"):
        print(f"Skipping run {run!s}")
        continue
    name = run.name[len("run-"):]
    arity, _, rest = name.partition("-")
    if arity == "multi":
        rest, _, with_or_without = rest.rpartition("-")
    rest, _, cls_or_reg = rest.rpartition("-")
    column = rest or None
    print(f"Found run with arity {arity}, cls_or_reg {cls_or_reg}, and column {column}")
    pairings = defaultdict(lambda: {"gold": [], "pred": []})
    with (run / "3e-05" / "preds.tsv").open() as predf, Path("workdata/corpus/test.tsv").open() as goldf:
        golds = cycle(csv.DictReader(goldf, delimiter="\t"))
        preds = csv.reader(predf, delimiter="\t")
        for gold, (p_task, _p_id, p_val) in zip(golds, preds):
            p_column, _, p_cls_or_reg = p_task.rpartition("_")
            if p_column == "emo":
                p_column = "emotion"
            pairings[p_task]["gold"].append(map_gold(gold[p_column]))
            pairings[p_task]["pred"].append(map_pred(p_val, column=p_column, cls_or_reg=p_cls_or_reg))
    metrics = {}
    for task_name, task_data in pairings.items():
        metric = {}
        # here I'm pretty much just trying to closely reproduce the val_metrics.json
        # not all of these metrics are actually used, I think?
        if "_cls" in task_name:
            p, r, f, _ = precision_recall_fscore_support(
                task_data["gold"],
                task_data["pred"],
                average="macro",
            )
            metric["major"] = f
            p, r, f, _ = precision_recall_fscore_support(
                task_data["gold"],
                task_data["pred"],
                average="micro",
            )
            a = accuracy_score(task_data["gold"], task_data["pred"])
            metric["minor"] = {
                "precision": p.mean(),
                "recall": r.mean(),
                "f1": f.mean(),
                "acc": a,
            }
        elif "_reg" in task_name:
            corr_s, _ = spearmanr(task_data["gold"], task_data["pred"])
            corr_p, _ = pearsonr(task_data["gold"], task_data["pred"])
            metric["minor"] = {
                "pearson": corr_p,
                "spearmanr": corr_s,
                "corr": (corr_p + corr_s) / 2,
            }
            metric["major"] = metric["minor"]["corr"]
        else:
            raise ValueError(f"Unknown kind of task (neither cls nor reg?): {task_name}")
        metrics[task_name] = {"metrics": metric}
    with (run / "3e-05" / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=4)

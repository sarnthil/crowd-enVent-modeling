import json
from pathlib import Path

BASE_PATH = Path("workdata/tasks/configs")

appraisals = [
    "suddenness",
    "familiarity",
    "predict_event",
    "pleasantness",
    "unpleasantness",
    "goal_relevance",
    "chance_responsblt",
    "self_responsblt",
    "other_responsblt",
    "predict_conseq",
    "goal_support",
    "urgency",
    "self_control",
    "other_control",
    "chance_control",
    "accept_conseq",
    "standards",
    "social_norms",
    "attention",
    "not_consider",
    "effort",
]

for kind in ("cls", "reg"):
    for task_name in ["emo", *appraisals]:
    # classifiers:
        data = {
            "task": f"crowdenvent_{task_name}_{kind}",
            "paths": {
                "train": "../../../sources/crowd-enVent-train.tsv",
                "test":  "../../../sources/crowd-enVent-test.tsv",
                "val":  "../../../sources/crowd-enVent-val.tsv"
            },
            "name": f"{task_name}_{kind}"
        }
        with (BASE_PATH / f"{task_name}_{kind}_config.json").open("w") as f:
            json.dump(data, f)

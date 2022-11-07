import sys
import json
import pathlib

BASE_PATH = pathlib.Path("workdata/runs")

# ls workdata/runs/run-single-goal-relevance-reg/3e-05/
# 1643269630  args.json  best_model.metadata.json  best_model.p  done_file  test_preds.p  val_metrics.json  val_preds.p
for run in BASE_PATH.iterdir():
    if not run.name.startswith("run-"):
        continue
    for lr in run.iterdir():
        if not (lr / "done_file").exists():
            continue
        if not (lr / "test_metrics.json").exists():
            continue
        with (lr / "test_metrics.json").open() as f:
            data = json.load(f)
        data["meta"] = {
            "lr": lr.name,
            "run": run.name,
            "multi": "-multi-" in run.name,
            "single": "-single-" in run.name,
            "reg": run.name.endswith("-reg") or "-reg-" in run.name,
            "cls": run.name.endswith("-cls") or "-cls-" in run.name,
            "with-emo": (
                None if "-single-" in run.name
                else "without" not in run.name
            ),
        }
        json.dump(data, sys.stdout)
        print()

import csv
from pathlib import Path
from collections import defaultdict

seeds = [int(p.name.split("-")[-1].strip(".tsv")) for p in Path("preds").glob("*arity-reg-*.tsv")]

with Path("../../workdata/crowd-enVent_validation_deduplicated.tsv").open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    text_ids = [row["text_id"] for row in reader]

for type in "reg", "cls":
    for seed in seeds:
        data = defaultdict(list)
        for file in Path("preds").glob(f"run-single-*-{type}-{seed}.tsv"):
            column = file.name[len("run-single-"):].split(f"-{type}-")[0].replace("-", "_")
            with file.open() as f:
                for line in f:
                    data[column].append(line.split()[-1])
        with open("TAmodel_{}_{}.tsv".format({"reg": "regressor", "cls": "classifier"}[type], seed), "w") as f:
            writer = csv.DictWriter(f, fieldnames=["text_id", *data.keys()], delimiter="\t")
            writer.writeheader()
            for i in range(len(data["predict_event"])):  # :(
                writer.writerow({
                    "text_id": text_ids[i],
                    **{
                        key: data[key][i]
                        for key in data
                    }
                })

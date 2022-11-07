# Create a file just like the original test.tsv, but with predicted appraisals.
from pathlib import Path
from contextlib import ExitStack
import csv

def get_value(line):
    # chance_responsblt_cls   test-11 0
    return "1" if line.strip().endswith("0") else "5"

with ExitStack() as stack:
    files = {}
    for file in Path("preds").glob("run-single-*-cls.tsv"):
        appraisal = file.name[len("run-single-"):-len("-cls.tsv")].replace("-", "_")
        files[appraisal] = stack.enter_context(file.open())

    appraisals = [*files.keys()]

    original = stack.enter_context((Path("sources") / "crowd-enVent_validation_deduplicated.tsv").open())
    copy = stack.enter_context((Path("workdata") / "crowd-enVent-test-preds.tsv").open("w"))
    original = csv.DictReader(original, delimiter="\t")
    copy = csv.DictWriter(copy, fieldnames=original.fieldnames, delimiter="\t")
    copy.writeheader()

    for row in original:
        for appraisal in appraisals:
            old = row[appraisal]
            new = get_value(next(files[appraisal]))
            row[appraisal] = new
        copy.writerow(row)

# create gold from lead!
import csv
from pathlib import Path
from contextlib import ExitStack

# we read the real gold, and replace each appraisal value with the corresponding value from preds from:
# workdata/preds/run-single-{appraisal}-reg.tsv

# first of all, define a function that maps [0,1] to inty [1,5]s:
def scale_up(fraction):
    # this isn't 100% correct, but very close:
    return round(float(fraction) * 4 + 1)


# extract the appraisal from the path of a tsv file:
def appraisal_from_path(path):
    # run-single-other-responsblt-reg.tsv -> other_responsblt
    return path.name[len("run-single-"):-len("-reg.tsv")].replace("-", "_")


TRUE_GOLD = Path("workdata/corpus/crowd-enVent_validation_deduplicated.tsv")
PREDICTED_GOLD = Path("workdata/corpus/crowd-enVent_predicted_gold.tsv")

PRED_PATH = Path("workdata/preds")

with ExitStack() as stack:
    in_file = csv.DictReader(stack.enter_context(TRUE_GOLD.open()), delimiter="\t")
    out_file = csv.DictWriter(stack.enter_context(PREDICTED_GOLD.open("w")), delimiter="\t", fieldnames=in_file.fieldnames)
    out_file.writeheader()
    appraisal_files = {appraisal_from_path(path): stack.enter_context(path.open()) for path in PRED_PATH.glob("run-single-*-reg.tsv")}
    for row in in_file:
        for appraisal, file in appraisal_files.items():
            row[appraisal] = scale_up(next(file).split("\t")[-1].strip())
        out_file.writerow(row)

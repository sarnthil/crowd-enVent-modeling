# find instances where some but not all models predict correctly
import json
import csv
from contextlib import ExitStack

import click


def prediction_files_callback(ctx, option, values):
    # list of (label, filename)
    with ExitStack() as stack:
        files = {label: stack.enter_context(open(filename)) for label, filename in values}
        while True:
            try:
                yield {
                    label: json.loads(next(files[label]))
                    for label in files
                }
            except StopIteration:
                break


@click.command()
@click.option("--prediction-file", "-p", "prediction_files", nargs=2, multiple=True, callback=prediction_files_callback)
def cli(prediction_files):
    printed_header = False
    with open("workdata/crowd-enVent_validation_deduplicated.tsv") as gold:
        gold = csv.DictReader(gold, delimiter="\t")
        for gold, predictions in zip(gold, prediction_files):
            if not printed_header:
                print("id","gold", *predictions.keys(), "sentence", sep=" & ", end="\\\\\n")
                printed_header = True
            p_values = {prediction["label"] for prediction in predictions.values()}
            g_value = gold["emotion"]
            if g_value not in p_values or {g_value} == p_values:
                # all wrong or all right; boring
                continue
            print(gold["text_id"], g_value, *(prediction["label"] for prediction in predictions.values()), gold["hidden_emo_text"], sep=" & ", end="\\\\\n")


if __name__ == '__main__':
    cli()

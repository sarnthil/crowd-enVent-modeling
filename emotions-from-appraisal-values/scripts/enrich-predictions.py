import csv

import click


@click.command()
@click.option(
    "--gold", "-g", default="sources/crowd-enVent_validation_deduplicated.tsv"
)
@click.option("--output", "-o", type=click.File("w"))
@click.argument("preds", type=click.File())
def cli(preds, gold, output):
    with open(gold) as f:
        reader = csv.DictReader(f, delimiter="\t")
        writer = csv.DictWriter(
            output,
            fieldnames=["text_id", "original_emotion", "predicted_emotion"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row, prediction in zip(reader, preds):
            writer.writerow(
                {
                    "text_id": row["text_id"],
                    "original_emotion": row["emotion"],
                    "predicted_emotion": prediction.strip(),
                }
            )


if __name__ == "__main__":
    cli()

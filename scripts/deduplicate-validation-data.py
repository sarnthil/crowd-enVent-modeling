import csv
import click

@click.command()
@click.option("--generation", type=click.File("r"))
@click.option("--validation", type=click.File("r"))
@click.option("--output", "-o", type=click.File("w"))
def cli(generation, validation, output):
    text_ids = set()
    reader = csv.DictReader(validation, delimiter="\t")
    for row in reader:
        text_ids.add(row["text_id"])

    reader = csv.DictReader(generation, delimiter="\t")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        if row["text_id"] in text_ids:
            writer.writerow(row)

if __name__ == "__main__":
    cli()

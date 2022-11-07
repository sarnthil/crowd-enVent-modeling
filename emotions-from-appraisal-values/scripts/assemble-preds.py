import csv
from contextlib import ExitStack
from pathlib import Path

import click

@click.command()
@click.option("--type", "-t", type=click.Choice(["cls", "reg"]), default="cls")
@click.option("--input", "-i", type=click.Path(), required=True)
@click.option("--truth", type=click.File("r"), required=True)
@click.option("--output", "-o", type=click.File("w"), required=True)
def cli(type, input, output, truth):
    output.write("row")
    reader = csv.DictReader(truth, delimiter="\t")
    with ExitStack() as stack:
        files = list(Path(input).glob(f"*-{type}.tsv"))
        for file in files:
            key = file.name[len("run-single-"):-len("-cls.tsv")].replace("-", "_")
            output.write(f"\t{key}")
        output.write("\temotion\n")
        files = [stack.enter_context(file.open()) for file in files]
        for i, (gold, *lines) in enumerate(zip(reader, *files)):
            output.write(str(i))
            for line in lines:
                value = line.split("\t")[-1].strip()
                output.write(f"\t{value}")
            output.write(f"\t{gold['emotion']}")
            output.write("\n")


if __name__ == '__main__':
    cli()

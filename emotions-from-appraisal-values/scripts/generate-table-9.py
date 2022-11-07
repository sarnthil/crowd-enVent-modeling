import csv
import subprocess
from collections import defaultdict
from statistics import mean, stdev

import click

EMOTIONS = {
    "anger": "Anger",
    "boredom": "Boredom",
    "disgust": "Disgust",
    "fear": "Fear",
    "guilt": "Guilt",
    "joy": "Joy",
    "no-emotion": "No-emotion",
    "pride": "Pride",
    "relief": "Relief",
    "sadness": "Sadness",
    "shame": "Shame",
    "surprise": "Surprise",
    "trust": "Trust",
    "macro-avg": "Macro avg.",
}

COMMANDS = {
    "discretized-pred": [
        "python",
        "scripts/cli.py",
        "-v",
        "evaluate",
        "--bin",
        "sources/crowd-enVent_predicted_gold.tsv",
        "--model",
        "workdata/binned-$seed.joblib",
    ],
    "discretized-gold": [
        "python",
        "scripts/cli.py",
        "-v",
        "evaluate",
        "--bin",
        "sources/crowd-enVent_validation_deduplicated.tsv",
        "--model",
        "workdata/binned-$seed.joblib",
    ],
    "scaled-pred": [
        "python",
        "scripts/cli.py",
        "-v",
        "evaluate",
        "--scale",
        "sources/crowd-enVent_predicted_gold.tsv",
        "--model",
        "workdata/scaled-$seed.joblib",
    ],
    "scaled-gold": [
        "python",
        "scripts/cli.py",
        "-v",
        "evaluate",
        "--scale",
        "sources/crowd-enVent_validation_deduplicated.tsv",
        "--model",
        "workdata/scaled-$seed.joblib",
    ],
}


def seeded_command(command, seed):
    return [element.replace("$seed", seed) for element in command]


def to_nice_int(floaty_string):
    _zero, _, rest = floaty_string.partition(".")
    return int(rest)


def extract_data(output):
    lines = iter(output.replace(" avg", "-avg").split("\n"))
    for line in lines:
        if "Results:" in line:
            next(lines)
            break
    # consumed the header stuff, now we only have interesting lines:
    result = {}
    skipped_middle = False
    for line in lines:
        if not line.strip():
            if not skipped_middle:
                next(lines)
                line = next(lines)  # skip accuracy
                skipped_middle = True
            else:
                break
        emotion, precision, recall, fscore, support = line.split()
        result[emotion] = {
            "precision": to_nice_int(precision),
            "recall": to_nice_int(recall),
            "f1-score": to_nice_int(fscore),
        }
    return result


def aggregate(results):
    # seed -> emotion -> metric -> value
    emotions = list(results.values())[0].keys()
    metrics = list(list(results.values())[0].values())[0].keys()
    result = {}
    for emotion in emotions:
        result[emotion] = {
            metric: {
                "mean": mean(results[seed][emotion][metric] for seed in results),
                "stdev": stdev((results[seed][emotion][metric])/100 for seed in results),
            }
            for metric in metrics
        }
    return result



def print_header(order, appendix):
    if appendix:
        print(r"""\begin{tabular}{lcccc}
    \toprule
    """)
    else:
        print(r"""\begin{tabular}{lcccccccccccc}
    \toprule""")
    print("    ", end="")
    for i, group in enumerate(order, start=1):
        kind = group[0].split("-")[0].title()
        if appendix:
            print(r" & \multicolumn{2}{c}{%s (%d)} &" % (kind, i), end="")
        else:
            print(r" & \multicolumn{6}{c}{%s (%d)} &" % (kind, i), end="")
    print(r"\\")
    for group in order:
        for part in group:
            if appendix:
                print(r"& \AEmodel%s" % part.split("-")[1].title(), end="")
            else:
                print(r"& \multicolumn{3}{c}{\AEmodel%s}" % part.split("-")[1].title(), end="")
        if not appendix:
            print(r"& $\Delta$", end="")
    print(r"\\")
    if appendix:
        print(r"""
    \cmidrule(r){2-2}\cmidrule(r){3-3}\cmidrule(r){4-4}\cmidrule(r){5-5}
    Emotion & \F & \F & \F & \F \\
    \cmidrule(r){1-1}
    \cmidrule(r){2-2}\cmidrule(r){3-3}\cmidrule(r){4-4}\cmidrule(r){5-5}
"""[1:-1])
    else:
        print(r"""
    \cmidrule(r){2-4}\cmidrule(r){5-7}\cmidrule(r){8-8}\cmidrule(r){9-11}\cmidrule(r){12-14}\cmidrule(r){15-15}
    Emotion & P & R & \F & P & R & \F & \F & P & R & \F & P & R & \F & \F\\
    \cmidrule(r){1-1}
    \cmidrule(r){2-4}\cmidrule(r){5-7}\cmidrule(r){8-8}\cmidrule(r){9-11}\cmidrule(r){12-14}\cmidrule(r){15-15}
"""[1:-1])


@click.command()
@click.option(
    "--order",
    default="discretized-gold,discretized-pred;scaled-gold,scaled-pred",
    callback=lambda ctx, option, value: [
        group.split(",") for group in value.split(";")
    ],
)
@click.option("--seeds", default="1234,2142,42,23,512")
@click.option("--appendix", is_flag=True)
def cli(order, seeds, appendix):
    seeds = seeds.split(",")
    results = defaultdict(dict)
    for group in order:
        for command in group:
            for seed in seeds:
                process = subprocess.run(seeded_command(COMMANDS[command], seed), capture_output=True)
                results[command][seed] = extract_data(
                    process.stderr.decode(
                        "utf-8"
                    )
                )
    results = {command: aggregate(result) for command, result in results.items()}

    print_header(order, appendix)
    for emotion, emotion_name in EMOTIONS.items():
        if emotion == "macro-avg":
            if appendix:
                print(r"    \cmidrule(r){1-1}\cmidrule(r){2-2}\cmidrule(r){3-3}\cmidrule(r){4-4}\cmidrule(r){5-5}")
            else:
                print(r"    \cmidrule(r){1-1}\cmidrule(r){2-4}\cmidrule(r){5-7}\cmidrule(r){8-8}\cmidrule(r){9-11}\cmidrule(r){12-14}\cmidrule(r){15-15}")
        print(f"    {emotion_name}", end="")
        for group in order:
            results_a, results_b = results[group[0]][emotion], results[group[1]][emotion]
            if appendix:
                print(
                    "",
                    fr"\sd{{{results_a['f1-score']['mean']:1.0f}}}{{{results_a['f1-score']['stdev']:+.2f}}}",
                    fr"\sd{{{results_b['f1-score']['mean']:1.0f}}}{{{results_a['f1-score']['stdev']:+.2f}}}",
                    sep=" & ",
                    end="",
                )
            else:
                print(
                    "",
                    *(f"${results_a[measure]['mean']:1.0f}$" for measure in ("precision", "recall", "f1-score")),
                    *(f"${results_b[measure]['mean']:1.0f}$" for measure in ("precision", "recall", "f1-score")),
                    f"${results_a['f1-score']['mean'] - results_b['f1-score']['mean']:+1.0f}$",
                    sep=" & ",
                    end="",
                )
        print(r"\\")
    print("    \\bottomrule\n\\end{tabular}")

if __name__ == '__main__':
    cli()

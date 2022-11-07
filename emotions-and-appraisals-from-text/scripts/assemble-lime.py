import json
from collections import defaultdict, Counter
from statistics import mean
from pathlib import Path

scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
counts = defaultdict(Counter)
N = Counter()

for model in Path("../workdata/").glob("classi*"):
    _, y_value, *__ = model.name.split("-")
    interpret_file = model / "lime-explanation.json"
    if not interpret_file.exists():
        continue
    with interpret_file.open() as g:
        for line in g:
            data = json.loads(line.strip())
            # 20boxes \u0120had \u0120been \u0120delivered \u0120by \u0120my
            # \u0120father \u0120in \u0120law \u0120for \u0120a \u0120thousand
            # \u0120can ning \u0120jars \u0120that \u0120we \u0120didn 't
            # \u0120want . </s>", "gold_label": "anger", "explanations":
            # {"anger": [["t", -0.015629517219144826], [
            gold = data["gold_label"]
            for token, saliency in data["explanations"][gold]:
                # optional: stripping "new-word" marker
                # token = token.lstrip("\u0120")
                scores[y_value][gold][token].append(saliency)
                counts[y_value][token] += 1
            N[y_value] += 1


def scorer(saliencies, count, total):
    if False:
        return sum(saliencies)
    elif False:
        return mean(saliencies)
    return sum(saliencies) * (total / count)


with open("assembled-lime.json", "w") as f:
    json.dump(
        {
            y_value: {value: [
                tok
                for _, tok in
                sorted(((scorer(scores[y_value][value][token], counts[y_value][token], N[y_value]), token) for token in scores[y_value][value]), reverse=True)[:30]
            ] for value in scores[y_value]} for y_value in scores
        },
        f,
    )

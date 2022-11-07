import json
from collections import defaultdict, Counter
from statistics import mean
from pathlib import Path

scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
counts = defaultdict(Counter)
N = Counter()

for model in Path("../workdata/").glob("classi*"):
    _, y_value, *__ = model.name.split("-")
    interpret_file = model / "interpret-integrated.txt"
    if not interpret_file.exists():
        continue
    with interpret_file.open() as g:
        for line in g:
            value, _, tokens = line.strip().partition("\t")
            tokens = json.loads(tokens)
            for token, saliency in tokens:
                # optional: stripping "new-word" marker
                # token = token.lstrip("\u0120")
                scores[y_value][value][token].append(saliency)
                counts[y_value][token] += 1
            N[y_value] += 1


def scorer(saliencies, count, total):
    if False:
        return sum(saliencies)
    elif False:
        return mean(saliencies)
    return sum(saliencies) * (total / count)


with open("assembled-interpretation.json", "w") as f:
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

# instance we care about from the emo
# pride   [["<s>", 0.055885332491293155], ["An", 0.006723520979445637], ["\u0120art", 0.12866686936461172], ["\u0120piece", 0.08600023441506956], ["\u0120I", 0.006376351271893149], ["\u0120produced", 0.03431759018183651], ["\u0120was", 0.06994299745263055], ["\u0120auction", 0.015069689917975037], ["ed", 0.03027634547271119], ["\u0120off", 0.03085704487853331], ["\u0120for", 0.03533973831757789], ["\u0120charity", 0.0060384112345492295], ["\u0120and", 0.01425662937378626], ["\u0120raised", 0.019736149980646826], ["\u0120a", 0.02710559977030344], ["\u0120good", 0.002952880891416815], ["\u0120sum", 0.023171522420413454], ["\u0120for", 0.03196857645420672], ["\u0120the", 0.041421529490363616], ["\u0120cause", 0.013360830556913126], ["</s>", 0.3205321664764681]]
# i want  out:
# ('Appraisalname', text, [list of saliencies]),
# ('Appraisalname', text, [list of saliencies]), etc.

import json
from csv import DictReader
from pathlib import Path

def get_saliencies(tokens):
    last = None
    for token, saliency in tokens:
        if token == "<s>":
            yield saliency
            continue
        if last:
            if token.startswith("\u0120") or token == "</s>":
                yield last
                last = saliency
            else:
                last += saliency
        else:
            last = saliency
    yield last

def barplotted_TeXt(tokens):
    for token, saliency in tokens:
        if token.startswith("\u0120"):
            yield " "
            token = token[1:]
        yield rf"\bptext{{{token}}}{{{saliency}}}"


def clamp(value):
    return {"1": "low", "2": "low", "3": "low", "4": "high", "5": "high"}.get(value, value)


print(r"""\begin{table*}[t]
  \caption{Comparison of saliency maps extracted from the appraisal models for the same test example, where the height of the bar indicates the amount of impact a certain token has on the output prediction of the model.}
    \centering
    %\begin{tabular}{lp{0.6\textwidth}}
    \begin{tabular}{lccc}""")
print(r"\toprule")
print(r"Appraisal & Gold & Predicted & Running example \\")
print(r"\midrule")
for model in Path("../workdata/").glob("classi*"):
    _, y_value, *__ = model.name.split("-")
    interpret_file = model / "interpret-integrated.txt"
    if not interpret_file.exists():
        continue
    with interpret_file.open() as p, Path("../sources/crowd-enVent-test.tsv").open() as g:
        g_reader = DictReader(g, delimiter="\t")
        for p_line, g_line in zip(p, g_reader):
            truth = clamp(g_line[y_value])
            value, _, tokens = p_line.strip().partition("\t")
            tokens = json.loads(tokens)
            joined_text = "".join(token.replace("\u0120", " ") for token, _ in tokens)

            if "When I was told my mom had awoken from her coma" not in joined_text:
                continue

            print(y_value, truth, value, end=" & ", sep=" & ")
            print(*barplotted_TeXt(tokens), sep="", end="\\\\\n")
print(r"\bottomrule")
print(r"""\end{tabular}
\label{tab:saliency-maps}
\end{table*}""")

import json
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def where(**kwargs):
    return kwargs

def matching(row, filter_):
    for key, should_value in filter_.items():
        is_value = row["meta"][key.replace("_", "-")]
        if should_value != is_value:
            return False
    return True

def get_metrics(*filters, **perma_filter):
    with open("data/test_metrics.jsonl") as f:
        if not filters:
            filters = [{}]
        for filter_ in filters:
            matched_one = False
            f.seek(0)
            for line in f:
                data = json.loads(line)
                if not matching(data, perma_filter):
                    continue
                if matching(data, filter_):
                    matched_one = True
                    yield data
            if not matched_one:
                raise ValueError(f"No line matched filter {filter_!r}")


*results ,= get_metrics(single=True, lr="3e-05", cls=True)
data = []
for result in results:
    x = result["meta"]["run"][len("run-single-"):][:-len("-cls")]
    key = x.replace("-", "_") + "_cls"
    y = result[key]["metrics"]["minor"]["f1"]
    data.append((x, y, "fuckoff"))

df = pd.DataFrame(data, columns=["Classifiers", "F1", "fuckoff"])
plot = sns.barplot(x="Classifiers", y="F1", hue="fuckoff", data=df, ci=None)
plot.legend_.remove()
plot.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.autoscale()
figure = plot.get_figure()
figure.set_size_inches(10, 7)
figure.savefig(f"plot-simple-cls.pdf", bbox_inches="tight")
plt.clf()


*results ,= get_metrics(single=True, lr="3e-05", reg=True)
data = []
for result in results:
    x = result["meta"]["run"][len("run-single-"):][:-len("-reg")]
    key = x.replace("-", "_") + "_reg"
    y = result[key]["metrics"]["minor"]["spearmanr"]
    data.append((x, y, "fuckoff"))

df = pd.DataFrame(data, columns=["Regressors", "Spearman", "fuckoff"])
plot = sns.barplot(x="Regressors", y="Spearman", hue="fuckoff", data=df, ci=None)
plot.legend_.remove()
plot.tick_params(axis="x", rotation=90)
figure = plot.get_figure()
figure.set_size_inches(10, 7)
figure.savefig(f"plot-simple-reg.pdf", bbox_inches="tight")
plt.clf()


multi, *singles = get_metrics(where(multi=True, with_emo=True), where(single=True), lr="3e-05", cls=True)
data = []
for single in singles:
    x = single["meta"]["run"][len("run-single-"):][:-len("-cls")]
    key = x.replace("-", "_") + "_cls"
    y = multi[key]["metrics"]["minor"]["f1"] - single[key]["metrics"]["minor"]["f1"]
    data.append((x, y, "fuckoff"))
df = pd.DataFrame(data, columns=["Classifiers", "F1 delta", "fuckoff"])
plot = sns.barplot(x="Classifiers", y="F1 delta", hue="fuckoff", data=df, ci=None)
plot.legend_.remove()
plot.tick_params(axis="x", rotation=90)
figure = plot.get_figure()
figure.set_size_inches(10, 7)
figure.savefig(f"plot-multi-vs-single-with-emo.pdf", bbox_inches="tight")
plt.clf()

multi, *singles = get_metrics(where(multi=True, with_emo=False), where(single=True), lr="3e-05", cls=True)
data = []
for single in singles:
    if "emo-cls" in single["meta"]["run"]:
        continue
    x = single["meta"]["run"][len("run-single-"):][:-len("-cls")]
    key = x.replace("-", "_") + "_cls"
    y = multi[key]["metrics"]["minor"]["f1"] - single[key]["metrics"]["minor"]["f1"]
    data.append((x, y, "fuckoff"))
df = pd.DataFrame(data, columns=["Classifiers", "F1 delta", "fuckoff"])
plot = sns.barplot(x="Classifiers", y="F1 delta", hue="fuckoff", data=df, ci=None)
plot.legend_.remove()
plot.tick_params(axis="x", rotation=90)
figure = plot.get_figure()
figure.set_size_inches(10, 7)
figure.savefig(f"plot-multi-vs-single-without-emo.pdf", bbox_inches="tight")
plt.clf()

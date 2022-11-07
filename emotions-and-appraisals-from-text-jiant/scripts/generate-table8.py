import json
from statistics import mean

import click

HUMAN_SCORES = {
    "cls": {
        "suddenness": 72,
        "familiarity": 68,
        "predict_event": 68,
        "pleasantness": 87,
        "unpleasantness": 84,
        "goal_relevance": 70,
        "chance_responsblt": 68,
        "self_responsblt": 78,
        "other_responsblt": 75,
        "predict_conseq": 62,
        "goal_support": 75,
        "urgency": 61,
        "self_control": 67,
        "other_control": 71,
        "chance_control": 70,
        "social_norms": 75,
        "accept_conseq": 53,
        "standards": 71,
        "attention": 65,
        "not_consider": 68,
        "effort": 67,
    },
    "reg": {
        "suddenness": 53,
        "familiarity": 43,
        "predict_event": 44,
        "pleasantness": 77,
        "unpleasantness": 74,
        "goal_relevance": 47,
        "chance_responsblt": 42,
        "self_responsblt": 67,
        "other_responsblt": 57,
        "predict_conseq": 28,
        "goal_support": 61,
        "urgency": 32,
        "self_control": 44,
        "other_control": 47,
        "chance_control": 37,
        "social_norms": 58,
        "accept_conseq": 12,
        "standards": 57,
        "attention": 33,
        "not_consider": 49,
        "effort": 45,
    },
}

APPRAISALS = {
    "suddenness": "Suddenness",
    "familiarity": "Familiarity",
    "predict_event": "Predict. Event",
    "pleasantness": "Pleasantness",
    "unpleasantness": "Unpleasantness",
    "goal_relevance": "Goal Relev.",
    "chance_responsblt": "Resp. Chance",
    "self_responsblt": "Resp. Self",
    "other_responsblt": "Resp. Other",
    "predict_conseq": "Predict Conseq.",
    "goal_support": "Goal Support",
    "urgency": "Urgency",
    "self_control": "Control Self",
    "other_control": "Control Other",
    "chance_control": "Control Chance",
    "social_norms": "Social Norms",
    "accept_conseq": "Accept Conseq.",
    "standards": "Standards",
    "attention": "Attention",
    "not_consider": "Not Consider",
    "effort": "Effort",
}

with open("workdata/all_test_results_seeded_grouped.jsonl") as f:
    models = json.load(f)


print(r"""
\begin{tabular}{lrrrrrr}
    \toprule
    & \multicolumn{3}{c}{Classification (\F)} & \multicolumn{3}{c}{Regression ($\rho$)} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}
    Appraisal & $T2A_{\text{human}}$  & $T2A_{\text{model}}$ & $\Delta_{cls}$ & $T2A_{\text{human}}$ & $T2A_{\text{model}}$ & $\Delta_{reg}$  \\
    \cmidrule(lr){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}
"""[1:-1])

all_human_cls, all_human_reg, all_model_cls, all_model_reg = [], [], [], []
for appraisal, fancy_appraisal in APPRAISALS.items():
    human_cls, human_reg = HUMAN_SCORES["cls"][appraisal], HUMAN_SCORES["reg"][appraisal]
    model_cls = round(models[f"{appraisal}_cls"]["f1"]["mean"] * 100)
    model_cls_stdev = models[f"{appraisal}_cls"]["f1"]["stdev"]
    model_reg = round(models[f"{appraisal}_reg"]["pearson"]["mean"] * 100)
    model_reg_stdev = models[f"{appraisal}_reg"]["pearson"]["stdev"]
    print("   ", fancy_appraisal, end=" & $")
    print(
        human_cls,
        f"{model_cls} \\: ({model_cls_stdev:.3f})",
        human_cls - model_cls,
        human_reg,
        f"{model_reg} \\: ({model_reg_stdev:.3f})",
        human_reg - model_reg,
        sep="$ & $",
        end="$ \\\\\n",
    )
    all_human_cls.append(human_cls)
    all_human_reg.append(human_reg)
    all_model_cls.append(model_cls)
    all_model_reg.append(model_reg)

print(r"    \midrule")

print("    Average", end=" & $")
human_cls, human_reg = round(mean(all_human_cls)), round(mean(all_human_reg))
model_cls, model_reg = round(mean(all_model_cls)), round(mean(all_model_reg))
print(human_cls, model_cls, human_cls - model_cls, human_reg, model_reg, human_reg - model_reg, sep="$ & $", end="$ \\\\\n")

print(r"""
    \bottomrule
\end{tabular}
\label{tab:appraisal-single-vs-val-results}
""")

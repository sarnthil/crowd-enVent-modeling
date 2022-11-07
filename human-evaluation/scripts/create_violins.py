import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import click

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
    "accept_conseq": "Accept Conseq.",
    "standards": "Standards",
    "social_norms": "Social Norms",
    "attention": "Attention",
    "not_consider": "Not Consider",
    "effort": "Effort",
}

COLUMNS = list(APPRAISALS) + [
    "round_number",
    "emotion",
    "text_id",
    "timestamp",
    "prolific_id",
    "anger",
    "boredom",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "pride",
    "relief",
    "sadness",
    "shame",
    "surprise",
    "trust",
    "generated_text",
    "hidden_emo_text",
    "event_duration",
    "emotion_duration",
    "intensity",
    "confidence",
    "self_control",
    "previous_participation",
    "age",
    "gender",
    "education",
    "ethnicity",
    "extravert",
    "critical",
    "dependable",
    "anxious",
    "open",
    "quiet",
    "sympathetic",
    "disorganized",
    "calm",
    "conventional",
    "did_you_lie?",
    "original_demographics",
]

STEPS = ("generation", "validation")

# read data

dfs = {}
for step in STEPS:
    datapath = f"/Users/sarnthil/CEAT/crowdsourcing-appraisals/outputs/corpus/crowd-enVent_{step}.tsv"
    df = pd.read_csv(datapath, sep="\t", header=0)
    df = df.reset_index()  # adds an "index" column with row id
    dfs[step] = pd.melt(
        df, id_vars=["index"], value_vars=list(APPRAISALS)
    )  # to long format
    dfs[step]["step"] = step
df = dfs["generation"].append(dfs["validation"])
sns.set_theme(style="whitegrid")

figure, axes = plt.subplots(nrows=3, ncols=7, figsize=(11, 7))
for (appraisal, title), (i, j) in zip(
    APPRAISALS.items(),
    ((i, j) for i in range(3) for j in range(7)),
):
    plot = sns.violinplot(
        x="variable",
        y="value",
        hue="step",
        hue_order=STEPS,
        split=True,
        data=df[df["variable"] == appraisal],
        ax=axes[i, j],
        legend=False,
    )
    plot.set_facecolor("whitesmoke")
    axes[i, j].set_title(title, fontsize=8)
    axes[i, j].set(xlabel="", ylabel="")
    axes[i, j].xaxis.set_major_formatter(plt.NullFormatter())
    axes[i, j].legend([], [], frameon=False)
    axes[i, j].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[i, j].yaxis.set_major_formatter(ticker.ScalarFormatter())


plt.tight_layout()
fig = figure.get_figure()
fig.savefig("violins_gen_vs_val.pdf")

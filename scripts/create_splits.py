from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 3141592654  # nothing up my sleeve
SOURCES = Path("sources/")

validated_text_ids = set(pd.read_csv(SOURCES / "crowd-enVent_validation.tsv", sep="\t").text_id)

df = pd.read_csv(SOURCES / "crowd-enVent_generation.tsv", sep="\t")
df = df[~df['text_id'].isin(validated_text_ids)]

assert len(set(df.text_id)) == 5400, "text ids should be unique"

# sklearn train_test_split doesn't support more than 2 splits, so we have to
# call it twice. This results in 80:10:10 splits.
outputs = {}
outputs["train"], rest = train_test_split(df, test_size=0.2, random_state=SEED)
outputs["test"], outputs["val"] = train_test_split(rest, test_size=0.5, random_state=SEED)

for split, data in outputs.items():
    data.to_csv(SOURCES / f"crowd-enVent-{split}.tsv", sep="\t")

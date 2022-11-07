## What?

This folder helps in recreating the results for **Experiment (1)** illustrated in
our depiction linked in the main `README.md` (or Figure 10 in the paper).

That means we focus hear on evaluating how the readers (validators) predicted
the prompting emotions and the correct appraisals from text, treating these two
"human models" separately (i.e., **T→E human** and **T→A human**). Note that **(1)** acts
as an _upper bound_ for the automatic classifiers.

## How to use the code in this folder?
1. Make sure you the original corpus linked in `sources/`
2. Install dependencies by doing the following:
    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `python3 -m pip install -r requirements.txt`
3. Call `make` to reproduce the scores for the human models.
4. Call `python create_violins.py` if you want to reproduce the violins plot.

## What?

This folder helps in recreating the results for **Experiment (2) and Experiment (3)** illustrated in
our depiction linked in the main `README.md` (or Figure 10 in the paper).

## How to use the code in this folder?

1. Make sure you the original corpus linked in `sources/`
2. Install dependencies by doing the following:
    - `python3 -m venv venv`
    - `source venv/bin/activate`
    -  `python3 -m pip install -r requirements.txt`
3. Call `make train` to train the models.
4. Call `make evaluate` to evaluate the models.

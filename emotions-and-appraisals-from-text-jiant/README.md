## How to use this code

#### Steps

1. Prepare data as before and install the required packages in a venv:
    - Install dependencies by doing the following:
        - `python3 -m venv venv`
        - `source venv/bin/activate`
        - `python3 -m pip install -r requirements.txt`
2. Set up jiant by calling `make jiant`. This will clone jiant, checkout a specific version, and copy extra files from `jiant-scripts`.
3. Then call `make prepare` to prepare the task configs and the data.
4. To train a model, take a look at the `Makefile` and the examples shown under the `train` target.
5. Then do the required call to train the model you want, e.g.: `python3 crowdenvent_single.py emo_cls` for training an emotion classifier or call `python3 crowdenvent_single.py suddenness_cls` and
`python3 crowdenvent_single.py suddenness_reg` to train a classifier and regressor model for suddenness.
6. If you use a scheduling system like `slurm` then you can adapt the scripts for scheduling all experiments (`scripts/schedule-single.sh`)


## Why?

We decided to test `jiant` since it supports an easy set up for transfer and multitask learning experiments. We did not report our results on the multitask experiments in our paper, but if you are curious you can run them.

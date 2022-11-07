import csv
import logging
import random

import click
import joblib
import numpy as np
import torch
from torch import nn, optim, utils
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, classification_report

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger()


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

def get_lr(optimizer):
    """"
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']



class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_out),
        )
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)



    def fit(self, X_train, y_train):
        criterion = nn.CrossEntropyLoss()
        train_data = np.append(
            X_train.astype(np.float32),
            y_train.reshape(y_train.shape[0], 1).astype(np.float32),
            axis=1,
        )
        loader = DataLoader(train_data, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters()) #weight_decay=0.001,)# nesterov=True)
        # nn.utils.clip_grad_norm_(self.parameters(), 10)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(loader), epochs=100)
        # self.fit()

        for epoch in range(30):
            running_loss = 0
            for i, data in enumerate(loader):
                X, y = data[:, :-1], data[:, -1]

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(X)
                target = y.long()
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    print(
                        f"\r[{epoch:2d}:{i:5d}] loss: {running_loss / 10:.3f}",
                        end="",
                    )
                    running_loss = 0
            # scheduler.step()
            print()
        return self

    def _predict(self, X_test):
        outputs = self(torch.from_numpy(X_test).float())
        return outputs.argmax(axis=-1).numpy()

    def predict(self, X_test):
        inverse = {v: k for k, v in self.encoding.items()}
        return [inverse[val] for val in self._predict(X_test)]

    def score(self, X_test, y_test):
        predictions = self._predict(X_test)
        scores = f1_score(y_test, predictions, average="micro")
        scores = classification_report(y_test, predictions, target_names=self.encoding)
        return scores


def load_tsv(file, columns, *, map=lambda x: x):
    file.seek(0)
    reader = csv.DictReader(file, delimiter="\t")
    data = []
    for line in reader:
        if len(columns) == 1:
            data.append(map(line[columns[0]]))
        else:
            data.append(list(map(line[col]) for col in columns))
    return np.array(data)


@click.group()
@click.option("--verbose", "-v", count=True)
@click.option("--quiet", "-q", count=True)
def cli(verbose, quiet):
    log.setLevel(
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL][
            max(min(2 - verbose + quiet, 4), 0)
        ]
    )


def map_scale(value):
    return (int(value) - 1) / 4


def map_inverse(value):
    return 1.0 - float(value)


def map_bin(value):
    return 1.0 if int(value) >= 4 else 0.0


def encode(y_train, encoding=None):
    if encoding is None:
        values = sorted(set(y_train))
        encoding = {val: i for i, val in enumerate(values)}
    return np.array([encoding[val] for val in y_train]), encoding


@cli.command()
@click.argument("data", type=click.File("r"))
@click.option(
    "--optimizer", default="adam", type=click.Choice(["adam", "sgd", "lbfgs"])
)
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["relu", "identity", "tanh", "logistic"]),
)
@click.option("--max-iterations", default=1000, type=int)
@click.option("--output", required=True, type=click.File("wb"))
@click.option("--bin", is_flag=True)
@click.option("--inverse", is_flag=True)
@click.option("--scale", is_flag=True)
@click.option("--seed", default=1234)
def train(data, optimizer, activation, max_iterations, output, bin, inverse, scale, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.info("Loading training data...")
    map = map_bin if bin else map_scale if scale else map_inverse if inverse else float
    X_train = load_tsv(data, APPRAISALS.keys(), map=map)
    y_train = load_tsv(data, ["emotion"])
    y_train, encoding = encode(y_train)

    log.info("Training...")
    clf = MLP(n_in=X_train.shape[1], n_out=len(set(y_train))).fit(X_train, y_train)
    clf.encoding = encoding

    log.info("Dumping model...")
    joblib.dump(clf, output)


@cli.command()
@click.option("--model", type=click.File("rb"))
@click.argument("data", type=click.File("r"))
@click.option("--bin", is_flag=True)
@click.option("--inverse", is_flag=True)
@click.option("--scale", is_flag=True)
def evaluate(model, data, bin, inverse, scale):
    log.info("Loading model...")
    map = map_bin if bin else map_scale if scale else map_inverse if inverse else float
    clf = joblib.load(model)
    log.info("Loading test data...")
    X_test = load_tsv(data, APPRAISALS.keys(), map=map)
    y_test = load_tsv(data, ["emotion"])
    y_test, _ = encode(y_test, encoding=clf.encoding)

    log.info("Evaluating...")
    log.info("Results: %s", clf.score(X_test, y_test))


@cli.command()
@click.option("--model", type=click.File("rb"))
@click.option("--output", "-o", type=click.File("w"), required=True)
@click.option("--bin", is_flag=True)
@click.option("--inverse", is_flag=True)
@click.option("--scale", is_flag=True)
@click.argument("data", type=click.File("r"))
def predict(model, data, bin, inverse, scale, output):
    log.info("Loading model...")
    clf = joblib.load(model)
    log.info("Loading test data...")
    map = map_bin if bin else map_scale if scale else map_inverse if inverse else float
    X_test = load_tsv(data, APPRAISALS.keys(), map=map)
    y_test = load_tsv(data, ["emotion"])
    y_test, _ = encode(y_test, encoding=clf.encoding)

    log.info("Predicting...")
    for line in clf.predict(X_test):
        print(line, file=output)


if __name__ == "__main__":
    cli()

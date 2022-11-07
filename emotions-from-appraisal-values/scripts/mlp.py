import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils_data
import joblib
import logging
import csv

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, classification_report
import copy
import random
import time


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


def load_tsv(file, columns, *, map=lambda x: x):
    with open(file, "r") as f:
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        data = []
        for line in reader:
            if len(columns) == 1:
                data.append(map(line[columns[0]]))
            else:
                data.append(list(map(line[col]) for col in columns))
    return np.array(data)


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


def decode(labels, encoding):
    inverse = {v: k for k, v in encoding.items()}
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    return [inverse[val] for val in labels]



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 128)
        nn.init.kaiming_uniform_(self.input_fc.weight, mode='fan_in', nonlinearity='relu')
        self.hidden_fc = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.hidden_fc.weight, mode='fan_in', nonlinearity='relu')
        self.dropout =  nn.Dropout(0.5)
        self.output_fc = nn.Linear(64, output_dim)

        # for m in self.layers:
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred, h_2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    model.train()

    for i, data in enumerate(iterator):
        x, y = data[:, :-1], data[:, -1]

        x = x.to(device)
        y = y.to(device)

        target = y.long()
        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, target)

        acc = calculate_accuracy(y_pred, target)
        f1 = f1_score(y_pred.argmax(1, keepdim=True), target, average="macro")
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1.item()

    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        epoch_f1 / len(iterator),
    )


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.eval()

    with torch.no_grad():

        for i, data in enumerate(iterator):
            x, y = data[:, :-1], data[:, -1]

            x = x.to(device)
            y = y.to(device)
            target = y.long()

            y_pred, _ = model(x)

            loss = criterion(y_pred, target)

            acc = calculate_accuracy(y_pred, target)
            f1 = f1_score(y_pred.argmax(1, keepdim=True), target, average="macro")

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()

    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        epoch_f1 / len(iterator),
    )


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_predictions(model, iterator, device):

    model.eval()

    labels = []
    probs = []

    with torch.no_grad():

        for i, data in enumerate(iterator):
            x, y = data[:, :-1], data[:, -1]

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return labels, probs


def plot_confusion_matrix(labels, pred_labels, *, display_labels):

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=display_labels)
    cm.plot(values_format="d", cmap="Blues", ax=ax)
    plt.savefig("outputs/confusion-appraisals2emotion-cls.pdf", bbox_inches="tight")


def get_representations(model, iterator, device):

    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():

        for i, data in enumerate(iterator):
            x, y = data[:, :-1], data[:, -1]

            x = x.to(device)

            y_pred, h = model(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def plot_representations(data, labels, *, display_labels):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab20")
    handles, labels = scatter.legend_elements()
    ax.legend(handles, display_labels)
    plt.savefig("outputs/appraisals2emotion-tsne.pdf", bbox_inches="tight")


# output_pca_data = get_pca(outputs)
# plot_representations(output_pca_data, labels)

# intermediate_pca_data = get_pca(intermediates)
# plot_representations(intermediate_pca_data, labels)


def get_tsne(data, n_components=2):
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


# output_tsne_data = get_tsne(outputs)
# plot_representations(output_tsne_data, labels)

# intermediate_tsne_data = get_tsne(intermediates)
# plot_representations(intermediate_tsne_data, labels)

def cli():
    SEED = 1234
    VALID_RATIO = 0.9
    BATCH_SIZE = 32

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    log = logging.getLogger()

    train_data_file = "workdata/train-and-val.tsv"
    test_data_file = "sources/crowd-enVent-test.tsv"

    X_train = load_tsv(train_data_file, APPRAISALS.keys(), map=map_bin)
    y_train = load_tsv(train_data_file, ["emotion"])
    y_train, encoding = encode(y_train)

    X_test = load_tsv(test_data_file, APPRAISALS.keys(), map=map_bin)
    y_test = load_tsv(test_data_file, ["emotion"])
    y_test, _ = encode(y_test, encoding)

    train_data = np.append(
        X_train.astype(np.float32),
        y_train.reshape(y_train.shape[0], 1).astype(np.float32),
        axis=1,
    )
    test_data = np.append(
        X_test.astype(np.float32),
        y_test.reshape(y_test.shape[0], 1).astype(np.float32),
        axis=1,
    )

    n_train_examples = int(.8 * X_train.shape[0])
    n_val_examples = X_train.shape[0] - n_train_examples
    train_data, valid_data = utils_data.random_split(
        train_data, [n_train_examples, n_val_examples]
    )

    valid_data = copy.deepcopy(valid_data)

    train_iterator = utils_data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_iterator = utils_data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_iterator = utils_data.DataLoader(test_data, batch_size=BATCH_SIZE)

    INPUT_DIM = X_train.shape[1]
    OUTPUT_DIM = 13

    model = MLP(INPUT_DIM, OUTPUT_DIM)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, ) #weight_decay=0.001,)# nesterov=True)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    ### TRAIN
    EPOCHS = 50

    best_valid_loss = float("inf")

    for epoch in trange(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc, train_f1 = train(
            model, train_iterator, optimizer, criterion, device
        )
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "model.pt")

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train F1: {train_f1*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1*100:.2f}%")

    model.load_state_dict(torch.load("model.pt"))

    test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion, device)

    print(f"Test Loss: {test_loss:.3f} | Test F1: {test_f1*100:.2f}%")

    ### EVAL
    labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)

    ### PLOTS
    plot_confusion_matrix(labels, pred_labels, display_labels=encoding.keys())

    outputs, intermediates, labels = get_representations(model, train_iterator, device)
    output_tsne_data = get_tsne(outputs)
    plot_representations(output_tsne_data, labels, display_labels=encoding.keys())

if __name__ == '__main__':
    cli()

from tensorboardX import SummaryWriter

import numpy as np
import torch
import networkx as nx
import dgl
import torch.nn.functional as F
import time
from torch.optim import SparseAdam, AdamW, SGD, Adam
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
# from Classification import NodeDataset

from GAT import GAT
from DatasetProcess import dataset
from Dataset import *


class EarlyStopping:
    def __init__(self, config, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.config = config

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # loss decreased, save model
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.config.save_path)


def accuracy(logits, labels):

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def f1_accuracy(logits, labels):

    _, indices = torch.max(logits, dim=1)
    f1_score = metrics.f1_score(indices, labels, average="macro")
    return f1_score


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def f1_evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return f1_accuracy(logits, labels)


def GATTrainer(config,
               dataset_name,
               num_classes=5,
               num_heads=8,
               num_out_heads=1,
               num_layers=1, # 1
               num_hidden=8, # 8
               in_drop=0.1,
               attn_drop=0.1,
               negative_slope=0.05,
               residual=False,
               early_stop=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # dataset defined in DatasetProcess(dgl.graph)
    cur_dataset = dataset(dataset_name)
    graph = cur_dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = num_classes

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    n_edges = cur_dataset.graph.number_of_edges()
    heads = ([num_heads] * num_layers) + [num_out_heads]

    model = GAT(graph,
                num_layers,
                num_feats,
                num_hidden,
                n_classes,
                heads,
                F.elu,
                in_drop,
                attn_drop,
                negative_slope,
                residual)

    if early_stop:
        stopper = EarlyStopping(patience=200, config=config)
    loss_fcn = torch.nn.CrossEntropyLoss()

    optim = Adam(model.parameters(), lr=float(config.lr), weight_decay=5e-4)

    dur = []
    for epoch in range(config.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask].long())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = f1_accuracy(logits[train_mask], labels[train_mask])

        val_acc = f1_evaluate(model, features, labels, val_mask)
        if early_stop:
            if stopper.step(val_acc, model):
                break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    if early_stop:
        model.load_state_dict(torch.load(config.save_path))
    acc = evaluate(model, features, labels, test_mask)
    f1_acc = f1_evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    print("Test f1 Accuracy {:.4f}".format(f1_acc))


def GATClassification(config,
                      dataset_name,
                      num_classes=5,
                      num_heads=8,
                      num_out_heads=1,
                      num_layers=1, # 1
                      num_hidden=8, # 8
                      in_drop=0.1,
                      attn_drop=0.1,
                      negative_slope=0.05,
                      residual=False,
                      early_stop=False):
    writer = SummaryWriter('runs/gatExps', comment="GAT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # dataset defined in DatasetProcess(dgl.graph)
    cur_dataset = dataset(dataset_name)
    graph = cur_dataset[0]

    features = graph.ndata['feat']
    labels = graph.ndata['label']
    test_mask = graph.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = num_classes

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    n_edges = cur_dataset.graph.number_of_edges()
    heads = ([num_heads] * num_layers) + [num_out_heads]

    model = GAT(graph,
                num_layers,
                num_feats,
                num_hidden,
                n_classes,
                heads,
                F.elu,
                in_drop,
                attn_drop,
                negative_slope,
                residual)

    params = torch.load(config.save_path)  # 加载参数
    model.load_state_dict(params)

    acc = evaluate(model, features, labels, test_mask)
    f1_acc = f1_evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    print("Test f1 Accuracy {:.4f}".format(f1_acc))

    embedding = model(features)
    writer.add_embedding(embedding, metadata=labels, global_step=0, tag=dataset_name)


if __name__ == "__main__":

    # configs
    actor_config = ConfigClass(save_path="./hw3/out/actor/actor_GAT_ckpt")
    chameleon_config = ConfigClass(save_path="./hw3/out/chameleon/chameleon_GAT_ckpt")
    cora_config = ConfigClass(save_path="./hw3/out/cora/cora_GAT_ckpt")

    # pretrain
    # GATTrainer(actor_config, "actor", num_classes=5, early_stop=True) # cora:7, others:5
    # GATTrainer(chameleon_config, "chameleon", num_classes=5, early_stop=True)
    # GATTrainer(cora_config, "cora", num_classes=7, early_stop=True)

    # results with pretrained models
    print("GAT：")
    GATClassification(actor_config, "actor", num_classes=5, early_stop=True)  # cora:7, others:5
    GATClassification(chameleon_config, "chameleon", num_classes=5, early_stop=True)
    GATClassification(cora_config, "cora", num_classes=7, early_stop=True)

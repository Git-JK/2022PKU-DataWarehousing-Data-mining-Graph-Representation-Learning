import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv
from sklearn import metrics


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        f1_score = metrics.f1_score(indices, labels, average="macro")
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), f1_score

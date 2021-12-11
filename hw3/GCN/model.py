import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, graph, in_feats, n_hidden, n_class, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats=in_feats, out_feats=n_hidden, activation=activation))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(in_feats=n_hidden, out_feats=n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(in_feats=n_hidden, out_feats=n_class))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.graph, h)
        return h
    
    # def hidden_representation(self, features):
    #     h = features
    #     with torch.no_grad():
    #         for i, layer in enumerate(self.layers):
    #             h = layer(self.graph, h)
    #         return h
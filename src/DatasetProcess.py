import dgl
from dgl.data import DGLDataset
import torch
import os
import numpy as np
import pandas as pd
from torch._C import Graph

class CoraDataset(DGLDataset):
    '''
    Turn cora dataset into DGLgraph
    '''
    def __init__(self):
        super().__init__(name='cora')
    
    def process(self):
        '''
        read in edges to get all src nodes and dst nodes, and read in features, labels, masks to form a DGLgraph
        '''
        edges_data = pd.read_csv('./dataset/cora/edge_list.csv')
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
        features = []
        for line in open('./dataset/cora/feature.txt', 'r').readlines():
            line = line.strip('\n').split(' ')
            for i in range(len(line)):
                line[i] = int(line[i])
            features.append(line)
        node_features = torch.from_numpy(np.array(features))
        labels = []
        for line in open('./dataset/cora/label.txt', 'r').readlines():
            line = int(line.strip('\n'))
            labels.append(line)
        node_labels = torch.from_numpy(np.array(labels))
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=2708)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label']= node_labels
        n_nodes = 2708
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_labels = []
        for line in open('./dataset/cora/train_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            train_labels.append(line)
        val_labels = []
        for line in open('./dataset/cora/val_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            val_labels.append(line)
        test_labels = []
        for line in open('./dataset/cora/test_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            test_labels.append(line)
        for i in train_labels:
            train_mask[i] = True
        for i in val_labels:
            val_mask[i] = True
        for i in test_labels:
            test_mask[i] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph
    def __len__(self):
        return 1

class ChameleonDataset(DGLDataset):
    '''
    Turn chameleon dataset into DGLgraph
    '''
    def __init__(self):
        super().__init__(name='chameleon')
    
    def process(self):
        '''
        read in edges to get all src nodes and dst nodes, and read in features, labels, masks to form a DGLgraph
        '''
        edges_data = pd.read_csv('./dataset/chameleon/edge_list.csv')
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
        features = []
        for line in open('./dataset/chameleon/feature.txt', 'r').readlines():
            line = line.strip('\n').split(' ')
            for i in range(len(line)):
                line[i] = int(line[i])
            features.append(line)
        node_features = torch.from_numpy(np.array(features))
        labels = []
        for line in open('./dataset/chameleon/label.txt', 'r').readlines():
            line = int(line.strip('\n'))
            labels.append(line)
        node_labels = torch.from_numpy(np.array(labels))
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=2277)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label']= node_labels
        n_nodes = 2277
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_labels = []
        for line in open('./dataset/chameleon/train_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            train_labels.append(line)
        val_labels = []
        for line in open('./dataset/chameleon/val_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            val_labels.append(line)
        test_labels = []
        for line in open('./dataset/chameleon/test_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            test_labels.append(line)
        for i in train_labels:
            train_mask[i] = True
        for i in val_labels:
            val_mask[i] = True
        for i in test_labels:
            test_mask[i] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph
    def __len__(self):
        return 1

class ActorDataset(DGLDataset):
    '''
    Turn actor dataset into DGLgraph
    '''
    def __init__(self):
        super().__init__(name='actor')
    
    def process(self):
        '''
        read in edges to get all src nodes and dst nodes, and read in features, labels, masks to form a DGLgraph
        '''
        edges_data = pd.read_csv('./dataset/actor/edge_list.csv')
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
        features = []
        for line in open('./dataset/actor/feature.txt', 'r').readlines():
            line = line.strip('\n').split(' ')
            for i in range(len(line)):
                line[i] = int(line[i])
            features.append(line)
        node_features = torch.from_numpy(np.array(features))
        labels = []
        for line in open('./dataset/actor/label.txt', 'r').readlines():
            line = int(line.strip('\n'))
            labels.append(line)
        node_labels = torch.from_numpy(np.array(labels))
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=7600)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label']= node_labels
        n_nodes = 7600
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_labels = []
        for line in open('./dataset/actor/train_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            train_labels.append(line)
        val_labels = []
        for line in open('./dataset/actor/val_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            val_labels.append(line)
        test_labels = []
        for line in open('./dataset/actor/test_nodes.txt', 'r').readlines():
            line = int(line.strip('\n'))
            test_labels.append(line)
        for i in train_labels:
            train_mask[i] = True
        for i in val_labels:
            val_mask[i] = True
        for i in test_labels:
            test_mask[i] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph
    def __len__(self):
        return 1
    
def dataset(name):
    '''
    a function to return the graph of the dataset whose name is the input
    '''
    if name == "cora":
        return CoraDataset()
    elif name == 'chameleon':
        return ChameleonDataset()
    elif name == "actor":
        return ActorDataset()
    else:
        raise ValueError("dataset name should be explicit")
    
if __name__ == "__main__":
    print(CoraDataset()[0])
    print(ChameleonDataset()[0])
    print(ActorDataset()[0])
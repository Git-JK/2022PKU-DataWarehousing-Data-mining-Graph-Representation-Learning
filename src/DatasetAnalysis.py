import torch
from DatasetProcess import CoraDataset, ChameleonDataset, ActorDataset
import dgl
import numpy as np
from dgl.data import DGLDataset
import matplotlib.pyplot as plt

def DataAnalyse(dataset):
    graph = dataset[0]
    num_nodes = graph.num_nodes()
    in_degree = graph.in_degrees()
    out_degree = graph.out_degrees()
    true_degree = in_degree + out_degree
    avg_dgree = torch.sum(true_degree) / num_nodes
    min_degree = int(torch.min(true_degree))
    max_degree = int(torch.max(true_degree))
    histdata = []
    for i in true_degree:
        histdata.append(int(i))
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title('probability of degree' + '(' + dataset.name + ')')
    plt.hist(histdata, bins= np.arange(min_degree, max_degree, 5), density=True)
    plt.savefig('./result/' + dataset.name + '.png')
    
    

DataAnalyse(CoraDataset())
    
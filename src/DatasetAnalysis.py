import torch
from DatasetProcess import CoraDataset, ChameleonDataset, ActorDataset
import numpy as np
import matplotlib.pyplot as plt

from dgl.data import DGLDataset
# import torch_geometric
import dgl
import networkx as nx
from dgl.data import DGLDataset
import scipy.sparse as sp
from torch.utils.data import DataLoader
import pickle

import time


edge_list = []


def read_txt_neighbor(file_path=""):
    neighbor = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            n1, n2 = list(map(int, line.strip().split(" ")[:2]))
            if n1 not in neighbor:
                neighbor[n1] = [n2]
            elif n2 not in neighbor[n1]:
                neighbor[n1].append(n2)
    return neighbor


# 求平均聚集系数
def get_average_CE(neighbor):
    ce_list = []

    for key in neighbor:
        node_set = set()
        edge_set = set()
        for node in neighbor[key]:
            node_set.add(node)
        for node in node_set:
            for neighborNode in neighbor[node]:
                if neighborNode in node_set:
                    edge_set.add((node, neighborNode))
        neighborNodeNum = len(node_set)
        neighborEdgeNum = len(edge_set) / 2

        ceNum = 0
        if neighborNodeNum > 1:
            ceNum = ceNum + 2 * neighborEdgeNum / ((neighborNodeNum-1) * neighborNodeNum)
        # print(ceNum)
        ce_list.append(ceNum)
        node_set.clear()
        edge_set.clear()
    # print(ce_list)
    total_ce = 0
    for ce in ce_list:
        total_ce += ce
    return total_ce / len(ce_list)


# node number: actor 7600, chameleon 2277, cora 2707
# edge number: actor 53318, chameleon 62742, cora 10556
# 请注意edge_list.txt文件中无向边已由两条有向边的形式给出
# 度数总和=edge number
def DataAnalyse(dataset):

    graph = dataset[0]
    num_nodes = graph.num_nodes()
    in_degree = graph.in_degrees()
    out_degree = graph.out_degrees()
    true_degree = in_degree
    avg_degree = torch.sum(true_degree) / num_nodes
    print('Average degree of ' + dataset.name + f' is {avg_degree}')

    neighbor = read_txt_neighbor('./dataset/' + dataset.name + '/edge_list.txt')
    avg_ce = get_average_CE(neighbor)
    print('Average Convergence Factor of ' + dataset.name + f' is {avg_ce}')

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
DataAnalyse(ChameleonDataset())
DataAnalyse(ActorDataset())


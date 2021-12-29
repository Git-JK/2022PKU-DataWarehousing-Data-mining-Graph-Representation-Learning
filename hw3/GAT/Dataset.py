import random
from networkx.classes.function import nodes
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl

from decimal import *
import collections
from tqdm import tqdm


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()


def makeDist(graph, graphpath="", power=0.75):

    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0

    # print(graph)

    for l in range(graph.num_edges()):
        src = graph.edges()[0][l]
        dst = graph.edges()[1][l]
        weight = 1

        edgedistdict[tuple([src, dst])] = weight
        nodedistdict[src] += weight

        weightsdict[tuple([src, dst])] = weight
        nodedegrees[src] += weight

        weightsum += weight
        negprobsum += np.power(weight, power)

    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    for edge, weight in edgedistdict.items():
        edgedistdict[edge] = weight / weightsum

    return edgedistdict, nodedistdict, weightsdict, nodedegrees


def negSampleBatch(sourcenode, targetnode, negsamplesize, weights,
                   nodedegrees, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:
        samplednode = nodesaliassampler.sample_n(1)
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        else:
            negsamples += 1
            yield samplednode


def makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler):
    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = []
        for negsample in negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        weights, nodedegrees, nodesaliassampler):
            for node in negsample:
                negnodes.append(node)
        yield [e[0], e[1]] + negnodes


class NodesDataset(Dataset):
    '''
    return nodes of a graph
    '''
    def __init__(self, nodes):
        self.nodes = nodes
    def __len__(self):
        return len(self.nodes)
    def __getitem__(self, index):
        return self.nodes[index]
    
class CollateFunction(object):
    '''
    collate function set, including how to random walk, configuration and how to sample pairs for skip-gram model
    '''
    def __init__(self, graph, config, walk_mode="random_walk"):
        # init parameters and which graph is used and the walk mode
        self.walk_mode = config.walk_mode
        self.p = config.p
        self.q = config.q
        self.walk_length = config.walk_length
        self.half_win_size = config.win_size // 2
        self.walk_num_per_node = config.walk_num_per_node
        self.graph = graph
        self.neg_num = config.neg_num
        self.nodes = graph.nodes().tolist()
    
    def sample_walks(self, graph, seed_nodes, walk_length, walk_mode):
        # according to the graph, starting nodes, walk length and walk mode, return random walks
        if walk_mode == "random_walk":
            walks = dgl.sampling.random_walk(graph, seed_nodes, length=walk_length)
        elif walk_mode == "node2vec_random_walk":
            walks = dgl.sampling.node2vec_random_walk(graph, seed_nodes, self.p, self.q, walk_length=walk_length)
        else:
            raise ValueError('walk mode should be explicit')
        return walks
    
    def skip_gram_gen_pairs(self, walk, half_win_size=2):
        # according to the walk path, generate pairs for skip gram model
        src = []
        dst = []
        l = len(walk)
        for i in range(l):
            real_win_size = half_win_size
            left = max(i - real_win_size, 0)
            right = i + real_win_size
            if right >= l:
                right = l - 1
            for j in range(left, right + 1):
                if walk[i] == walk[j]:
                    continue
                src.append(walk[i])
                dst.append(walk[j])
        return src, dst
    
    def __call__(self, batch_nodes):
        # sample random walks and generate pairs [src, dst](a batch) as skip-gram model's input from the walk paths
        batch_src = []
        batch_dst = []
        walk_list = []
        for i in range(self.walk_num_per_node):
            walks = self.sample_walks(self.graph, batch_nodes, self.walk_length, self.walk_mode)
            walk_list += walks[0].tolist()
        for walk in walk_list:
            src, dst = self.skip_gram_gen_pairs(walk, self.half_win_size)
            batch_src += src
            batch_dst += dst
        
        batch_tmp = list(set(zip(batch_src, batch_dst)))
        random.shuffle(batch_tmp)
        batch_src, batch_dst = zip(*batch_tmp)
        
        batch_src = torch.from_numpy(np.array(batch_src))
        batch_dst = torch.from_numpy(np.array(batch_dst))
        return batch_src, batch_dst
    
class ConfigClass(object):
    def __init__(self, save_path, lossdata_path="", order = 1, lr=0.05,gpu="0"):
        self.lr = 0.02
        self.gpu = "0"
        self.epochs = 200
        self.embed_dim = 64
        self.batch_size = 10
        self.walk_num_per_node = 6
        self.walk_length = 12
        self.win_size = 6
        self.neg_num = 5
        self.walk_mode = "random_walk"
        self.p = 2
        self.q = 2
        self.save_path = save_path
        self.lossdata_path = lossdata_path
        self.order = 1
    
if __name__ == "__main__":
    # test whether the classes above work normally
    from DatasetProcess import CoraDataset
    graph = CoraDataset()[0]
    nodes_dataset = NodesDataset(graph.nodes())
    
    config = ConfigClass("./out/test.txt")
    pair_generate_func = CollateFunction(graph, config)
    pair_loader = DataLoader(nodes_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=pair_generate_func)
    pair = set()
    for i, (batch_src, batch_dst) in enumerate(pair_loader):
        print(batch_src.shape)
        print(batch_dst.shape)
        for i, j in zip(batch_src.tolist(), batch_dst.tolist()):
            pair.add((i, j))
        print(len(pair))
        break

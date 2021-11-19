import random
from networkx.classes.function import nodes
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl

class NodesDataset(Dataset):
    def __init__(self, nodes):
        self.nodes = nodes
    def __len__(self):
        return len(self.nodes)
    def __getitem__(self, index):
        return self.nodes[index]
    
class CollateFunction(object):
    def __init__(self, graph, config, walk_mode="random_walk"):
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
        if walk_mode == "random_walk":
            walks = dgl.sampling.random_walk(graph, seed_nodes, length=walk_length)
        elif walk_mode == "node2vec_random_walk":
            walks = dgl.sampling.node2vec_random_walk(graph, seed_nodes, self.p, self.q, walk_length=walk_length)
        else:
            raise ValueError('walk mode should be explicit')
        return walks
    
    def skip_gram_gen_pairs(self, walk, half_win_size=2):
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
    def __init__(self, save_path, lr=0.05,gpu="0"):
        self.lr = 0.005
        self.gpu = "0"
        self.epochs = 120
        self.embed_dim = 64
        self.batch_size = 10
        self.walk_num_per_node = 6
        self.walk_length = 12
        self.win_size = 6
        self.neg_num = 5
        self.walk_mode = "random_walk"
        self.p = 1.0
        self.q = 1.0
        self.save_path = save_path
    
if __name__ == "__main__":
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
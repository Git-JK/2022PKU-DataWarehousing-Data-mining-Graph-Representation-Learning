import time
import numpy as np
from tqdm import tqdm
import torch
import random
import dgl
from torch.optim import SparseAdam
from torch.utils.data import Dataset, DataLoader
from Model import SkipGramModel
from DatasetProcess import dataset
from Dataset import NodesDataset, ConfigClass


def skip_gram_gen_pairs(walk, half_win_size=2):
    src, dst = list(), list()

    l = len(walk)
    for i in range(l):
        real_win_size = half_win_size
        left = i - real_win_size
        if left < 0:
            left = 0
        right = i + real_win_size
        if right >= l:
            right = l - 1
        for j in range(left, right + 1):
            if walk[i] == walk[j]:
                continue
            src.append(walk[i])
            dst.append(walk[j])
    return src, dst


class Call_Func:
    def __init__(self, g, config):
        self.g = g
        self.p = config.p
        self.q = config.q
        self.walk_length = config.walk_length
        self.half_win_size = config.win_size // 2

    def __call__(self, nodes):
        batch_src, batch_dst = list(), list()

        walks_list = list()
        walks = dgl.sampling.node2vec_random_walk(self.g, nodes, p=1, q=1, walk_length=self.walk_length)
        walks_list += walks.tolist()
        for walk in walks_list:
            src, dst = skip_gram_gen_pairs(walk, self.half_win_size)
            batch_src += src
            batch_dst += dst

        # shuffle pair
        batch_tmp = list(zip(batch_src, batch_dst))
        random.shuffle(batch_tmp)
        batch_src, batch_dst = zip(*batch_tmp)

        batch_src = torch.from_numpy(np.array(batch_src))
        batch_dst = torch.from_numpy(np.array(batch_dst))
        return batch_src, batch_dst


def node2vecTrainer(config, dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    graph = dataset(dataset_name)[0]

    model = SkipGramModel(graph.num_nodes(), embed_dim=config.embed_dim).to(device)

    optimizer = SparseAdam(model.parameters(), lr=float(config.lr))

    nodes_dataset = NodesDataset(graph.nodes())

    pair_generate_func = Call_Func(graph, config)

    pair_loader = DataLoader(nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                             collate_fn=pair_generate_func)

    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()

        loss_total = []
        top_loss = 100
        tqdm_bar = tqdm(pair_loader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (batch_src, batch_dst) in enumerate(tqdm_bar):
            batch_src = batch_src.cuda().long()
            batch_dst = batch_dst.cuda().long()

            batch_neg = np.random.randint(0, graph.num_nodes(), size=(batch_src.shape[0], config.neg_num))
            batch_neg = torch.from_numpy(batch_neg).cuda().long()

            model.zero_grad()
            loss = model.forward(batch_src, batch_dst, batch_neg, config.batch_size)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())

        if top_loss > np.mean(loss_total):
            top_loss = np.mean(loss_total)
            torch.save(model.state_dict(), config.save_path)
            print("Epoch: %03d; loss = %.4f saved path: %s" % (epoch, top_loss, config.save_path))
        print("Epoch: %03d; loss = %.4f cost time %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))


if __name__ == "__main__":
    dataset_name = ['actor', 'chameleon', 'cora']
    train_set = dataset_name[2]
    config = ConfigClass(save_path=f"./out/{train_set}/{train_set}_node2vec_ckpt")
    node2vecTrainer(config, train_set)

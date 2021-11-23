import torch
import dgl
import numpy as np
import random
import time
from tqdm import tqdm, trange
from torch.optim import SparseAdam, AdamW, SGD
from torch.utils.data import Dataset, DataLoader

from DatasetProcess import dataset
from Dataset import NodesDataset, CollateFunction, ConfigClass
from Dataset import *
from LineModel import Line


def lineTrainer(config, dataset_name, order=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # dataset defined in DatasetProcess(dgl.graph)
    graph = dataset(dataset_name)[0]

    model = Line(graph.num_nodes(), embed_dim=config.embed_dim, order=order).to(device)

    optimizer = SparseAdam(model.parameters(), lr=float(config.lr))
    # optimizer = AdamW(model.parameters(), lr=float(config.lr))

    # lossdata = {"it": [], "loss": []}
    # it = 0

    edgedistdict, nodedistdict, weights, nodedegrees = makeDist(
        graph
    )
    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    batch_range = int(len(edgedistdict) / config.batch_size)

    start_time = time.time()
    for epoch in range(config.epochs):
        # model.train()

        loss_total = []
        top_loss = 100
        # tqdm_bar = tqdm(pair_loader, desc="Training epoch{epoch}".format(epoch=epoch))
        for b in trange(batch_range):
            samplededges = edgesaliassampler.sample_n(config.batch_size)
            batch = list(makeData(samplededges, config.neg_num, weights, nodedegrees, nodesaliassampler))
            batch = torch.LongTensor(batch)
            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negsamples = batch[:, 2:]

            model.zero_grad()
            loss = model.forward(v_i, v_j, negsamples, config.batch_size, device)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())

        if top_loss > np.mean(loss_total):
            top_loss = np.mean(loss_total)
            torch.save(model.state_dict(), config.save_path)
            print(model.state_dict())
            print("Epoch: %03d; loss = %.4f saved path: %s" % (epoch, top_loss, config.save_path))
        print("Epoch: %03d; loss = %.4f cost time %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))


if __name__ == "__main__":

    # 一阶相似度
    # config = ConfigClass(save_path="./out/actor/actor_line_1_ckpt", lossdata_path="./out/actor/actor_line_loss.pkl")
    # lineTrainer(config, "actor")

    # config = ConfigClass(save_path="./out/chameleon/chameleon_line_1_ckpt")
    # lineTrainer(config, "chameleon")

    # config = ConfigClass(save_path="./out/cora/cora_line_1_ckpt")
    # lineTrainer(config, "cora")

    # 二阶相似度
    config = ConfigClass(save_path="./out/actor/actor_line_2_ckpt", lossdata_path="./out/actor/actor_line_loss.pkl")
    lineTrainer(config, "actor", order=2)

    config = ConfigClass(save_path="./out/chameleon/chameleon_line_2_ckpt")
    lineTrainer(config, "chameleon", order=2)

    config = ConfigClass(save_path="./out/cora/cora_line_2_ckpt")
    lineTrainer(config, "cora", order=2)

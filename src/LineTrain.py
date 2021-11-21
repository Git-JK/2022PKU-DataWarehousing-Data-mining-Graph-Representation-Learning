import torch
import dgl
import numpy as np
import random
import time
from tqdm import tqdm
from torch.optim import SparseAdam
from torch.utils.data import Dataset, DataLoader
import pickle
import sys

from DatasetProcess import dataset
from Dataset import NodesDataset, CollateFunction, ConfigClass
from Model import SkipGramModel
from LineModel import Line


def lineTrainer(config, dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # dataset defined in DatasetProcess(dgl.graph)
    graph = dataset(dataset_name)[0]

    # set[] of nodes in graph(can get with index)
    nodes_dataset = NodesDataset(graph.nodes())

    model = Line(graph.num_nodes()+1, embed_dim=config.embed_dim).to(device)
    # model = SkipGramModel(graph.num_nodes(), embed_dim=config.embed_dim).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.lr), momentum=0.9, nesterov=True)
    # optimizer = SparseAdam(model.parameters(), lr=float(config.lr))

    lossdata = {"it": [], "loss": []}
    it = 0

    pair_generate_func = CollateFunction(graph, config)

    pair_loader = DataLoader(nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                             collate_fn=pair_generate_func)

    start_time = time.time()
    for epoch in range(config.epochs):
        # model.train()

        # loss_total = []
        top_loss = 100

        tqdm_bar = tqdm(pair_loader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (batch_src, batch_dst) in enumerate(tqdm_bar):
            batch_src = batch_src.to(device).long()
            batch_dst = batch_dst.to(device).long()

            batch_neg = np.random.randint(0, graph.num_nodes(), size=(batch_src.shape[0], config.neg_num))
            batch_neg = torch.from_numpy(batch_neg).to(device).long()

            model.zero_grad()
            loss = model(batch_src, batch_dst, batch_neg, device)
            loss.backward()
            optimizer.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

            #loss = model.forward(batch_src, batch_dst, batch_neg, config.batch_size)
            # loss.backward()
            # optimizer.step()
            # loss_total.append(loss.detach().item())

        if top_loss > np.mean(lossdata["loss"]):
            top_loss = np.mean(lossdata["loss"])
            torch.save(model.state_dict(), config.save_path)
            print("Epoch: %03d; loss = %.4f saved path: %s" % (epoch, top_loss, config.save_path))
        print("Epoch: %03d; loss = %.4f cost time %.4f" % (epoch, np.mean(lossdata["loss"]), time.time() - start_time))
    print("\nDone training, saving model to {}".format(config.save_path))
    torch.save(model, "{}".format(config.save_path))

    print("Saving loss data at {}".format(config.lossdata_path))
    with open(config.lossdata_path, "wb") as ldata:
        pickle.dump(lossdata, ldata)
    sys.exit()


if __name__ == "__main__":
    config = ConfigClass(save_path="./out/actor/actor_line_ckpt", lossdata_path="./out/actor/actor_line_loss.pkl")
    lineTrainer(config, "actor")

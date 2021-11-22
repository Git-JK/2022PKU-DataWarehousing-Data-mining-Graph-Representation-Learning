import time
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SparseAdam
from torch.utils.data import Dataset, DataLoader
from Model import SkipGramModel
from DatasetProcess import dataset
from Dataset import NodesDataset, CollateFunction, ConfigClass


# def train(graph, model, config):
#     optimizer = SparseAdam(model.parameters(), lr=float(config.lr))
    
#     best_val_acc = 0
#     best_test_acc = 0
    
#     features = graph.ndata['feat']
#     labels = graph.ndata['label']
#     train_mask = graph.ndata['train_mask']
#     val_mask = graph.ndata['val_mask']
#     test_mask = graph.ndata['test_mask']

#     for epoch in range(config.epochs):
        


def deepwalkTrainer(config, dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True
    
    graph = dataset(dataset_name)[0]
    
    model = SkipGramModel(graph.num_nodes(), embed_dim=config.embed_dim).to(device)
    
    optimizer = SparseAdam(model.parameters(), lr=float(config.lr))
    
    nodes_dataset = NodesDataset(graph.nodes())
    
    pair_generate_func = CollateFunction(graph, config)
    
    pair_loader = DataLoader(nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=pair_generate_func)
    
    
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
            print("Epoch: %03d; loss = %.4f saved path: %s" %(epoch, top_loss, config.save_path))
        print("Epoch: %03d; loss = %.4f cost time %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))


if __name__ == "__main__":
    config = ConfigClass(save_path="./out/actor/actor_deepwalk_ckpt")
    deepwalkTrainer(config, "actor")

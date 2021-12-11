import time
import os
from model import GCN
from DatasetProcess2 import dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np
import dgl
import argparse

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args, mode):
    data = dataset(args.dataset)
    graph = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        graph = graph.int().to(args.gpu)
    
    features = graph.ndata['feat']
    labels = torch.tensor(graph.ndata['label'], dtype=torch.long)
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    val_mask = graph.ndata['val_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    
    if args.self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    
    n_edges = graph.number_of_edges()
    
    degs = graph.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    graph.ndata['norm'] = norm.unsqueeze(1)
    
    model = GCN(graph, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    
    if cuda:
        model.cuda()
    
    loss_fcn = CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    dur = []
    weight_path = "./hw3/out/" + args.dataset + "/" + args.dataset + "_gcn_ckpt"
    if mode == "test" and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    else:   
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                acc, n_edges / np.mean(dur) / 1000))
        
    print()
    save_path = "./hw3/out/" + args.dataset + "/" + args.dataset + "_gcn_ckpt"
    torch.save(model.state_dict(), save_path)
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'chameleon', 'actor').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight of L2 loss")
    parser.add_argument("--self-loop", action="store_true", help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)
    args = parser.parse_args()
    print(args)
    
    main(args, "test")
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args, load_data
from model import GraphSAGE, evaluate
from DatasetProcess2 import dataset
from tensorboardX import SummaryWriter


def main(args):
    writer = SummaryWriter('./runs/graphsageExps', comment="Graphsage")
    data = dataset(args.dataset)
    g = data[0]
    features = torch.tensor(g.ndata['feat'], dtype=torch.float)
    labels = torch.tensor(g.ndata['label'], dtype=torch.long)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc, f1_score = evaluate(model, g, features, labels, val_nid)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | F1 score {:.4f} |"
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, f1_score, n_edges / np.mean(dur) / 1000))

    print()
    acc, f1_score = evaluate(model, g, features, labels, test_nid)
    print("Test Accuracy {:.4f} | F1 score {:.2%}".format(acc, f1_score))
    save_path = "./hw3/out/" + args.dataset + "/" + args.dataset + "_graphsage_ckpt"
    torch.save(model.state_dict(), save_path)
    model.eval()

    embedding = model(g, features)
    writer.add_embedding(embedding, metadata=labels, global_step=0, tag=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--dataset", type=str, default="cora", help="cora or chameleon or actor")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn", help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)

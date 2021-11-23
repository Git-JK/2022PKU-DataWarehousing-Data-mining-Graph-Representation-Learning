from pandas._config import config
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import time
from torch.optim import AdamW
from tqdm import tqdm

from DatasetProcess import dataset
from LineModel import Line


class NodeDataset(Dataset):
    '''
    return a dataset for classification training/validation/testing, it consists of (node, node_label), label is one-hot vector
    '''
    def __init__(self, graph, mode="train"):
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
        label = graph.ndata["label"]
        label = label.numpy()
        nodes = graph.nodes().numpy()
        data = np.array([nodes, label]).T
        title = ['nodes', 'labels']
        df = pd.DataFrame.from_records(data, columns=title)
        # turn the labels into one-hot vectors
        df_label = pd.crosstab(df.nodes, df.labels).gt(0).astype(int)
        if mode == "train":
            self.nodes = np.array(graph.nodes()[train_mask])
        elif mode == "val":
            self.nodes = np.array(graph.nodes()[val_mask])
        elif mode == "test":
            self.nodes = np.array(graph.nodes()[test_mask])
        self.label = []
        for i in self.nodes:
            self.label.append(df_label.values[i])
        self.label = np.array(self.label)
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index], self.label[index]


class NodeClassification(nn.Module):
    '''
    use a fully connected layer and softmax for classification
    '''
    def __init__(self, emb, num_class=7):
        super(NodeClassification, self).__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=True)
        self.size = emb.shape[1]
        self.num_class = num_class
        self.fc = nn.Linear(self.size, self.num_class)
    
    def forward(self, node):
        node_emb = self.emb(node)
        prob = self.fc(node_emb)
        
        return prob


def evaluate(test_nodes_loader, model):
    # evaluate the classification model on the testing set, return macro f1 scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for i, (batch_nodes, batch_labels) in enumerate(test_nodes_loader):
        batch_nodes = batch_nodes.to(device).long()
        batch_labels = batch_labels.to(device).long()
        
        logit = model(batch_nodes)
        probs = torch.sigmoid(logit)
        label = torch.max(batch_labels.data, 1)[1].cpu().numpy()
        pred = torch.max(probs.data, 1)[1].cpu().numpy()
        
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, pred)

        # print("begin: \n labels:")
        # print(labels_all)
        # print("predict: ")
        # print(predict_all)
    f1_score = metrics.f1_score(labels_all, predict_all, average="macro")
    return f1_score


def classification(config, dataset_name):
    # classifier training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    os.environ["KERAS_BACKEND"]="pytorch"
    torch.backends.cudnn.benchmark = True

    # param = torch.load(config.save_path)
    param = torch.load(config.save_path, map_location='cpu')
    # print(param)

    emb = param['v_embeddings.weight']
    graph = dataset(dataset_name)[0]
    train_nodes_dataset = NodeDataset(graph, "train")
    test_nodes_dataset = NodeDataset(graph, "test")
    val_nodes_dataset = NodeDataset(graph, "val")
    # train_nodes_loader = DataLoader(train_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    # test_nodes_loader = DataLoader(test_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    # val_nodes_loader = DataLoader(val_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    train_nodes_loader = DataLoader(train_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_nodes_loader = DataLoader(test_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_nodes_loader = DataLoader(val_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    model = NodeClassification(emb, num_class=config.num_class)
    model.to(device)
    classifier_path = "./out/" + dataset_name + "/" + dataset_name + "_deepwalk_classification_ckpt"
    if os.path.exists(classifier_path):
        # model.load_state_dict(torch.load(classifier_path))
        model.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    else:
        optimizer = AdamW(model.parameters(), lr=float(config.lr), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        loss_func = nn.BCEWithLogitsLoss()
        
        start_time = time.time()
        for epoch in range(config.epochs):
            loss_total = []
            top_loss = 100
            model.train()
            tqdm_bar = tqdm(train_nodes_loader, desc="Training epoch{epoch}".format(epoch=epoch))
            for i, (batch_nodes, batch_labels) in enumerate(tqdm_bar):
                batch_nodes = batch_nodes.to(device).long()
                batch_labels = batch_labels.to(device).long()
                
                model.zero_grad()
                logit = model(batch_nodes)
                probs = F.softmax(logit, dim=1)
                loss = loss_func(probs, batch_labels.float())
                loss.backward()
                optimizer.step()
                
                loss_total.append(loss.detach().item())
            print("Training Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
            f1 = evaluate(val_nodes_loader, model)
            print("Validation Epoch: %03d; f1 = %.4f" %(epoch, f1))
            if top_loss > np.mean(loss_total):
                top_loss = np.mean(loss_total)
                torch.save(model.state_dict(), classifier_path)
                
    f1 = evaluate(test_nodes_loader, model)
    print("Testing dataset: f1 = %.4f"%(f1))


def line_classification(config, dataset_name, order=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    os.environ["KERAS_BACKEND"] = "pytorch"
    torch.backends.cudnn.benchmark = True

    # param = torch.load(config.save_path)
    param = torch.load(config.save_path, map_location='cpu')
    # print(param)

    graph = dataset(dataset_name)[0]
    train_nodes_dataset = NodeDataset(graph, "train")
    test_nodes_dataset = NodeDataset(graph, "test")
    val_nodes_dataset = NodeDataset(graph, "val")
    train_nodes_loader = DataLoader(train_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_nodes_loader = DataLoader(test_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_nodes_loader = DataLoader(val_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    if order==1:
        emb = param['nodes_embeddings.weight']
    else:
        emb = param['contextnodes_embeddings.weight']

    model = NodeClassification(emb, num_class=config.num_class)
    # model = Line(graph.num_nodes(), embed_dim=config.embed_dim).to(device)

    model.to(device)
    if order == 1:
        classifier_path = "./out/" + dataset_name + "/" + dataset_name + "_line_1_classification_ckpt"
    else:
        classifier_path = "./out/" + dataset_name + "/" + dataset_name + "_line_2_classification_ckpt"

    if os.path.exists(classifier_path):
        # model.load_state_dict(torch.load(classifier_path))
        model.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    else:
        optimizer = AdamW(model.parameters(), lr=float(config.lr), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        # optimizer = torch.optim.SGD(model.parameters(), lr=float(config.lr), momentum=0.9, nesterov=True)

        loss_func = nn.BCEWithLogitsLoss()

        start_time = time.time()
        for epoch in range(config.epochs):
            loss_total = []
            top_loss = 100
            model.train()
            tqdm_bar = tqdm(train_nodes_loader, desc="Training epoch{epoch}".format(epoch=epoch))
            for i, (batch_nodes, batch_labels) in enumerate(tqdm_bar):
                batch_nodes = batch_nodes.to(device).long()
                batch_labels = batch_labels.to(device).long()

                model.zero_grad()
                logit = model(batch_nodes)
                probs = F.softmax(logit, dim=1)
                loss = loss_func(probs, batch_labels.float())
                loss.backward()
                optimizer.step()

                loss_total.append(loss.detach().item())
            print("Training Epoch: %03d; loss = %.4f cost time  %.4f" % (
            epoch, np.mean(loss_total), time.time() - start_time))
            f1 = evaluate(val_nodes_loader, model)
            print("Validation Epoch: %03d; f1 = %.4f" % (epoch, f1))
            if top_loss > np.mean(loss_total):
                top_loss = np.mean(loss_total)
                torch.save(model.state_dict(), classifier_path)

                f1 = evaluate(test_nodes_loader, model)
                print("Testing dataset: f1 = %.4f" % (f1))

    f1 = evaluate(test_nodes_loader, model)
    print("Testing dataset: f1 = %.4f" % (f1))


if __name__ == "__main__":
    # train and test classifier on all datasets
    class ConfigClass():
        def __init__(self, save_path, num_class=5, lossdata_path=""):
            self.lr = 0.05
            self.gpu = "0"
            self.epochs = 100
            self.batch_size = 256
            self.num_class = num_class # cora:7, others:5
            self.save_path = save_path
            self.lossdata_path = lossdata_path

    # ---Deepwalk Model Classification
    print("Deepwalk Model:")
    config = ConfigClass("./out/actor/actor_deepwalk_ckpt")
    classification(config, "actor")
    # f1 = 0.3011

    # config = ConfigClass("./out/cora/cora_deepwalk_ckpt", num_class=7)
    # classification(config, "cora")
    # f1 = 0.6340

    # config = ConfigClass("./out/chameleon/chameleon_deepwalk_ckpt")
    # classification(config, "chameleon")
    # f1 = 0.5120

    # ---LINE Model Classification
    print("LINE Model:")

    # 一阶临近
    config = ConfigClass(save_path="./out/actor/actor_line_1_ckpt")
    line_classification(config, "actor")
    # f1 = 0.2980

    # config = ConfigClass(save_path="./out/cora/cora_line_1_ckpt", num_class=7)
    # line_classification(config, "cora")
    # f1 = 0.4335

    # config = ConfigClass(save_path="./out/chameleon/chameleon_line_1_ckpt")
    # line_classification(config, "chameleon")
    # f1 = 0.5169

    # 二阶临近
    config = ConfigClass(save_path="./out/actor/actor_line_2_ckpt")
    line_classification(config, "actor", order=2)
    # f1 = 0.2936

    # config = ConfigClass(save_path="./out/cora/cora_line_2_ckpt", num_class=7)
    # line_classification(config, "cora", order=2)
    # f1 = 0.4925

    # config = ConfigClass(save_path="./out/chameleon/chameleon_line_2_ckpt")
    # line_classification(config, "chameleon", order=2)
    # f1 = 0.5591

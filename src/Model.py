import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import loss

class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(SkipGramModel, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dim = embed_dim
        self.u_embeddings = nn.Embedding(num_nodes, embed_dim, sparse=True)
        self.v_embeddings = nn.Embedding(num_nodes, embed_dim, sparse=True)
        initrange = 0.5 / self.emb_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
    
    def forward(self, src, pos, neg, batch_size):
        embed_src = self.u_embeddings(src)
        embed_pos = self.v_embeddings(pos)
        embed_neg = self.v_embeddings(neg)
        
        score = torch.mul(embed_src, embed_pos)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()
        
        neg_score = torch.bmm(embed_neg, embed_src.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()
        
        loss = log_target + sum_log_sampled        
        return -1 * loss.sum() / batch_size


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkipGramModel(1000, embed_dim=32)
    model.to(device)
    src = np.random.randint(0, 100, size=10)
    src = torch.from_numpy(src).to(device).long()
    dst = np.random.randint(0, 100, size=10)
    dst = torch.from_numpy(dst).to(device).long()
    neg = np.random.randint(0, 100, size=(10, 5))
    neg = torch.from_numpy(neg).to(device).long()
    
    print(src.shape, dst.shape, neg.shape)
    print(model(src, dst, neg, 24))

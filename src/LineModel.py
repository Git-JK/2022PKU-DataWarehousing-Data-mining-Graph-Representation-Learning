import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Line(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order

        # self.nodes_embeddings = nn.Embedding(size, embed_dim)
        self.nodes_embeddings = nn.Embedding(size, embed_dim, sparse=True)

        if order == 2:

            # self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim, sparse=True)

            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(-.5, .5) / embed_dim

        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(-.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, batch_size, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1
        )
        loss = positivebatch + negativebatch
        # return -torch.mean(loss) / batch_size
        return -torch.mean(loss)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Line(1000, embed_dim=32)
    model.to(device)
    src = np.random.randint(0, 100, size=10)
    src = torch.from_numpy(src).to(device).long()
    dst = np.random.randint(0, 100, size=10)
    dst = torch.from_numpy(dst).to(device).long()
    neg = np.random.randint(0, 100, size=(10, 5))
    neg = torch.from_numpy(neg).to(device).long()

    print(src.shape, dst.shape, neg.shape)
    print(model(src, dst, neg, 24, device))

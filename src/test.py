from dgl.data import CoraGraphDataset

data = CoraGraphDataset()
g = data[0]
print(g.ndata['label'])
# 图数据挖掘 第三次作业

## 1 代码的运行方法

### 1.1 GCN model的运行

在**./hw3/GCN**目录命令行按如下要求输入指令运行即可：
```
python train.py + 参数


usage: train.py [-h] [--dataset DATASET] [--dropout DROPOUT] [--gpu GPU] [--lr LR] [--n-epochs N_EPOCHS] [--n-hidden N_HIDDEN] [--n-layers N_LAYERS] [--weight-decay WEIGHT_DECAY] [--self-loop]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name ('cora', 'chameleon', 'actor').
  --dropout DROPOUT     dropout probability
  --gpu GPU             gpu
  --lr LR               learning rate
  --n-epochs N_EPOCHS   number of training epochs
  --n-hidden N_HIDDEN   number of hidden gcn units
  --n-layers N_LAYERS   number of hidden gcn layers
  --weight-decay WEIGHT_DECAY
                        Weight of L2 loss
  --self-loop           graph self-loop (default=True)
```

若嫌麻烦，也可以直接在pycharm中直接运行**./hw3/GCN/train.py**,会代入默认参数进行运行

### 1.2 GAT model的运行

在**./hw3/GAT**目录下运行./hw3/GAT/GATTrain.py即可代入默认参数进行运行，其中包括训练过程以及对训练过后的模型的评估



## 2 各个Model的详细情况和效果对比

### 2.1 GCN

#### 2.1.1 实验过程和结果

- 数据集的处理和转化
    - 首先，为了能直接用pandas包更方便地读取一些数据，在**DataTransform.py**实现了一些对**edge_list.txt**的处理，转成了带列名的**edge_list.csv**
    - 然后，对每个dataset读入数据，在**DataProcess.py**处理成了**DGLgraph对象**,下面分别是cora,chameleon和actor转化后得到的DGLgraph
    ```
    Graph(num_nodes=2708, num_edges=10556,
      ndata_schemes={'feat': Scheme(shape=(1433,), dtype=torch.int32), 'label': Scheme(shape=(), dtype=torch.int32), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool)}
      edata_schemes={})
    Graph(num_nodes=2277, num_edges=62742,
        ndata_schemes={'feat': Scheme(shape=(2325,), dtype=torch.int32), 'label': Scheme(shape=(), dtype=torch.int32), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool)}
        edata_schemes={})
    Graph(num_nodes=7600, num_edges=53318,
        ndata_schemes={'feat': Scheme(shape=(932,), dtype=torch.int32), 'label': Scheme(shape=(), dtype=torch.int32), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool)}
        edata_schemes={})
    ```
- 对GCN框架下的图结点表示和分类训练
根据设置的隐藏层层数**n_hidden**,设置了**n_hidden**层**dgl**框架内置实现的**GraphConv**层，并根据设定的**dropout**进行默认**200 epochs**的训练
  
    - 用graph中的**train_mask**包含的结点作为训练数据，**val_mask**包含的结点作为每个训练epoch中的测试
    - 训练完后在**test_mask**包含的结点下进行分类测试，得到正确分类的比例**test accuracy、macro $F_1$ score**作为评测指标，下面从上到下分别为cora、chameleon、actor dataset下的**test accuracy、macro $F_1$ score**
    ```
    Test accuracy 79.20% | F1 score 79.06%
    Test accuracy 43.20% | F1 score 39.45%
    Test accuracy 23.90% | F1 score 8.19%
    ```
    - GCN model训练完后model参数也保存在了**./hw3/out/dataset_name**文件夹下，命名为dataset_name_gcn_ckpt。

### 2.2 GCN

#### 2.2.1 实验过程和结果

- 数据集的处理和转化

  - 这部分与之前其他模型一致，不再详细说明

- GAT框架以及训练过程

  - 对GAT框架下的图结点表示和分类训练

  - 设置使用的Attention Head数量**num_heads**，根据设置的隐藏层层数**num_hidden**,设置了**n_hidden**层**dgl**框架内置实现的**GraphConv**层，并根据设定的**dropout**进行默认**learning rate = 0.02**，**200 epochs**的训练

  - 用graph中的**train_mask**包含的结点作为训练数据，**val_mask**包含的结点作为每个训练epoch中的测试

  - 训练完后在**test_mask**包含的结点下进行分类测试，得到正确分类的比例**test accuracy、macro $F_1$ score**作为评测指标，下面从上到下分别为cora、chameleon、actor dataset下的**test accuracy、macro $F_1$ score**

    ```
    Test accuracy 77.40% | F1 score 77.53%
    Test accuracy 53.73% | F1 score 54.25%
    Test accuracy 24.12% | F1 score 12.63%
    ```

  - 上述训练结果的GAT model参数保存在了**./hw3/out/#dataset_name#**文件夹下，命名为#dataset_name#_GAT_ckpt。

### 2.4 GCN、GAT、GraphSage的效果对比

## 3 浅层模型和图神经网络模型的node embedding的可视化分析


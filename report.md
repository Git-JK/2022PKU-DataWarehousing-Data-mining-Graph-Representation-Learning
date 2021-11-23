# 图数据挖掘 第一次作业

## 1 代码的运行方法

### 1.0 数据分析任务的运行


### 1.1 Deepwalk Model的运行

1. 运行**DataTransform.py**
2. 运行**DataProcess.py**
3. 如果要进行重新训练model，运行**Deepwalk.py**
4. 运行**Classification.py**，在其最下方选择需要进行classification的数据集，注释掉对其他数据集分类的代码后运行，得到**macro $F_1$ score**

## 2 各个Model的详细情况

### 2.1 DeepWalk

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
- 对DeepWalk框架下的图结点表示的训练
在**Deepwalk.py**中对图上所有结点，每个结点进行**walk_num_per_node**次的random walk,在每次random walk的path中，在**win_size**大小的windows下选出输入skip-gram model的**[src, dst]** pair用于训练，skip-gram model在**Model.py**中实现，使用给定dataset生成的graph，以**SparseAdam**作为optimizer，对skip-gram model采用了negative sampling的训练方法以加快训练速度，进行**batch_size = 10, embed_dim = 64**的**120 epochs**的训练，最后得到的loss最低的model存在**./out/dataset_name**文件夹下的各个**deepwalk_ckpt**中
- 对已经生成的图结点的向量表示进行分类
在**Classification.py**中对已经被embed成向量的图结点进行分类任务的训练。
    - 首先读入各个结点的label，转化成**one-hot vector**的形式
    - 然后用一层全连接层加上softmax作为classifier，用graph中的**train_mask**包含的结点作为训练数据，**val_mask**包含的结点作为每个训练epoch中的测试
    - 训练完后在**test_mask**包含的结点下进行分类测试，得到**macro $F_1$ score**作为评测指标，下面从上至下分别为cora、chameleon、actor dataset下的**macro $F_1$ score**
    ```
    Testing dataset: f1 = 0.6340
    Testing dataset: f1 = 0.5120
    Testing dataset: f1 = 0.3011
    ```
    - classifier model训练完之后model也保存在了**./out/dataset_name**文件夹下，命名为dataset_name_deepwalk_classification_ckpt。
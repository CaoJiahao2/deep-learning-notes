# 图神经网络 (GNN)

> 图结构数据（社交网络、分子、知识图谱、推荐系统）是继序列之后最重要的数据形态。本章覆盖谱域与空间域两条技术主线、过平滑问题、图级任务与图基础模型。

---

## 目录

1. [概述：为什么需要 GNN](#1)
2. [图与任务的基本表示](#2)
3. [谱域方法：从频谱卷积到 GCN](#3)
4. [空间域方法：消息传递](#4)
5. [注意力与采样：GAT / GraphSAGE](#5)
6. [过平滑与深度困境](#6)
7. [图级任务：池化与图分类](#7)
8. [应用版图](#8)
9. [代码示例（PyG）](#9)
10. [实践建议](#10)
11. [相关文档](#11)
12. [参考文献](#12)

---

## 1 概述：为什么需要 GNN {#1}

图像是规则网格（CNN 擅长），文本是线性序列（RNN/Transformer 擅长），但大量真实数据是**不规则图**：

- 社交网络：用户—好友关系；
- 分子：原子—化学键；
- 知识图谱：实体—关系—实体三元组；
- 推荐系统：用户—商品二部图；
- 交通/电路/程序：天然图结构。

CNN/RNN 处理图有两个根本障碍：**排列不变性**（节点无固定顺序，模型输出不应随节点编号改变）与**可变拓扑**（邻居数量不定，卷积核无法固定尺寸）。GNN 的答案是：**在图上定义可学习的、排列不变的聚合算子**。

---

## 2 图与任务的基本表示 {#2}

### 2.1 图的定义

无向图 $G = (V, E)$，$|V| = n$ 个节点，每个节点带特征向量，得到特征矩阵：

$$X \in \mathbb{R}^{n \times d}, \quad A \in \{0, 1\}^{n \times n} \; \text{(邻接矩阵)}$$

度矩阵 $D_{ii} = \sum_j A_{ij}$。特征 $X$ 携带"是什么"，邻接 $A$ 携带"谁和谁有关系"。

### 2.2 三类任务

| 任务层次 | 定义 | 例子 |
|---|---|---|
| 节点级 | 预测每个节点的标签 | 社区检测、用户分类 |
| 边级 | 预测边的存在或属性 | 链接预测、关系抽取、推荐 |
| 图级 | 预测整图属性 | 分子性质、蛋白质功能 |

---

## 3 谱域方法：从频谱卷积到 GCN {#3}

### 3.1 图信号处理视角

把节点特征看作图上的信号 $x \in \mathbb{R}^n$。拉普拉斯矩阵 $L = D - A$（归一化 $L_{sym} = I - D^{-1/2} A D^{-1/2}$）是图上的"微分算子"，可特征分解：

$$L_{sym} = U \Lambda U^\top$$

特征向量 $U$ 构成"图傅里叶基"。谱域图卷积定义为频域相乘：

$$g_\theta \star x = U \, g_\theta(\Lambda) \, U^\top x$$

问题：$O(n)$ 特征分解不可扩展、基依赖具体图（无法跨图迁移）。

### 3.2 ChebNet 与 GCN

ChebNet 用 Chebyshev 多项式近似 $g_\theta(\Lambda)$，把卷积限制为 $K$ 阶局部化。取 $K = 1$ 并做重归一化技巧（加自环、对称归一化），得到 **GCN** 的逐层传播规则：

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right), \quad \tilde{A} = A + I$$

这就是"每个节点取自身与邻居特征的（度归一化）平均，再过线性变换与非线性"。GCN 简洁、有效，是 GNN 的 "ResNet 时刻"。

---

## 4 空间域方法：消息传递 {#4}

空间域视角更直观：GNN = 邻居间交换信息。每层执行三步：

```text
1. MESSAGE   邻居 j 生成消息 m_ij = φ(h_j, h_i, e_ij)
2. AGGREGATE 汇总邻居消息（必须排列不变）:
              h_i' = γ(h_i, AGG_{j∈N(i)} m_ij)
              AGG ∈ {sum, mean, max} 等对称函数
3. UPDATE    组合自身与聚合结果
```

**WL 同构测试**（Weisfeiler-Lehman）是消息传递 GNN 表达能力的理论上界：标准 MPNN 恰好等价于一阶 WL 测试，无法区分某些正则图。GIN（Graph Isomorphism Network）把聚合取为 sum（单射）达到该上界：

$$h_i^{(l+1)} = \text{MLP}^{(l)}\left((1+\epsilon) h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} h_j^{(l)}\right)$$

---

## 5 注意力与采样：GAT / GraphSAGE {#5}

### 5.1 GAT（Graph Attention Network）

邻居重要性不该一视同仁。GAT 用注意力权重替代固定归一化：

$$\alpha_{ij} = \frac{\exp\big(\text{LeakyReLU}(a^\top [W h_i \| W h_j])\big)}{\sum_{k \in \mathcal{N}(i)} \exp\big(\text{LeakyReLU}(a^\top [W h_i \| W h_k])\big)}$$

$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)$$

多头注意力 + 残差是标配。GAT 是"注意力机制在图上的回声"，也预示了 Transformer 与图结构的深刻联系（全连接图上的自注意力 = 全局 GAT）。

### 5.2 GraphSAGE（采样聚合）

大规模图的邻居数可达百万，全邻居聚合不可行。GraphSAGE 每次随机采样固定 $k$ 个邻居再聚合（mean/LSTM/pool），使 mini-batch 训练复杂度与图规模解耦，是工业界大规模图学习的基石。

### 5.3 方法谱系总结

| 方法 | 聚合方式 | 表达力 | 可扩展性 |
|---|---|---|---|
| GCN | 度归一化均值 | = 一阶 WL | 好 |
| GIN | 求和 MLP | = 一阶 WL（上界） | 好 |
| GAT | 注意力加权 | 自适应权重 | 中 |
| GraphSAGE | 采样聚合 | 取决于聚合器 | 极好（工业） |

---

## 6 过平滑与深度困境 {#6}

GCN 层数增加时性能不升反降，根源是**过平滑（oversmoothing）**：反复平均使所有节点表示收敛到不可分。

数学上，归一化邻接 $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 的谱半径为 1，多层传播 $\hat{A}^k X$ 把信号向主特征向量（全 1 向量）投影——$k \to \infty$ 时节点表示趋于一致。

缓解方案：

- **残差/初始残差**（GCNII）：$H^{(l+1)} = \sigma\big(\tilde{A} H^{(l)} + H^{(0)} W\big)$，保留初始信息的直通路径；
- **JK-Net**：跳跃连接融合各层表示；
- **重归一化/图谱滤波**：去掉低频过度平滑成分。

与 CNN 不同，GNN 的"深度"收益有限——**宽度（特征维度）与全局信息注入（虚拟节点、位置编码）通常比层数更有效**。Graph Transformer 路线（Graphormer 等）干脆用全注意力 + 结构/位置编码绕开层数困境。

---

## 7 图级任务：池化与图分类 {#7}

从节点表示得到图表示 $h_G$：

- **全局池化**：mean/max/sum pooling——简单但丢失结构；
- **DiffPool**：可学习的软聚类，层级聚合子图；
- **Set2Set / 注意力池化**：对节点集做注意力加权读出；
- **TOPKPool / SAGPool**：按分数保留 top-k 节点，构建层级粗化图。

分子性质预测的标准管线：原子特征（元素、价键、手性）→ 多层 GIN/GCN → 读出层 → MLP 回归/分类，在 QM9、HIV、PCBA 等 MoleculeNet 基准上评测。

---

## 8 应用版图 {#8}

### 8.1 推荐系统

用户—商品交互天然是二部图。PinSage（Pinterest，十亿级节点）把 GraphSAGE 用于工业推荐，随机游走定义重要性 + 采样聚合；LightGCN 用"仅线性传播 + 去除自环"的极简设计在协同过滤上稳定有效。图召回 + 排序模型是主流架构，详见[推荐系统与 CTR 模型](./recommendation.md)。

### 8.2 知识图谱

TransE 把实体/关系嵌入向量空间（$h + r \approx t$），GNN 版（CompGCN、R-GCN）用关系特定的变换聚合邻居。任务：链接预测（补全缺失三元组）、实体对齐。与 RAG 结合的知识图谱检索（GraphRAG）正在兴起，见[长上下文与检索增强](../llm-mllm/long-context-rag.md)。

### 8.3 科学计算

- **分子/药物**：GNN 是分子性质预测与虚拟筛选的主力（MoleculeNet、OGB-mol 基准）；等变 GNN（SchNet、DimeNet、EGNN）把平移/旋转对称性编码进网络，直接预测 3D 能量与力，用于分子动力学与蛋白-配体对接；
- **AlphaFold**：蛋白质折叠的 Evoformer 本质是带三角更新的图神经网络；
- **物理仿真**：GNS 用 GNN 学粒子动力学（流体、布料），是神经代理模型（neural surrogate）的代表。

### 8.4 其他

交通预测（时空图 + GNN/LSTM 混合）、芯片设计（Google 用 GNN 做布局布线强化学习）、代码分析（AST 即图）、欺诈检测（金融交易图）。

### 8.5 图与 LLM 的融合

- **GraphRAG**：把文档构建成知识图谱，检索时沿图扩展上下文，缓解长上下文的"大海捞针"；
- **图基础模型**：用 LLM 把节点属性转为文本描述，图结构作为提示（GPT4Graph、GraphGPT 系）；
- **LLM 做图推理**：基准（NLGraph 等）显示 LLM 在小图上尚可、大图结构推理仍是短板——图神经网络的结构归纳偏置短期内不可替代。

---

## 9 代码示例（PyG） {#9}

用 PyTorch Geometric 实现一个两层 GCN 做节点分类（Cora 引文网络）：

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]  # x: [2708, 1433], edge_index: [2, 10556]

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, training=self.training)
        return self.conv2(h, edge_index)

model = GCN(dataset.num_features, 16, dataset.num_classes)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()

model.eval()
pred = model(data).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"Test accuracy: {acc:.4f}")  # ~0.815
```

消息传递的核心抽象就在 `GCNConv(x, edge_index)` 一行：`edge_index` 是稀疏 COO 格式的边表，聚合沿边发生。

---

## 10 实践建议 {#10}

1. **层数宁浅勿深**：2–3 层几乎总是最优；需要远距离信息时优先考虑虚拟节点、全局注意力或位置编码，而非加深；
2. **聚合器选择**：可区分性优先选 sum（GIN）；对度敏感的任务用 mean/max 抑制度偏置；
3. **归一化很重要**：度归一化（GCN）或 BatchNorm/LayerNorm 在图上同样关键；
4. **基准先行**：OGB（Open Graph Benchmark）是当前标准评测，避免旧基准（Cora 类）的饱和与泄露问题；
5. **异构图要专门处理**：不同类型节点/边不能共享参数空间，用 R-GCN/HGT 类方法；
6. **大规模训练**：邻域采样（GraphSAGE 式）或图分区并行，DGL/PyG 均有工程支持。

---

## 11 相关文档 {#11}

- 注意力机制的源头：[Transformer](./transformer.md)
- 推荐系统中的图模型：[推荐系统与 CTR 模型](./recommendation.md)
- 知识图谱检索增强：[长上下文与检索增强](../llm-mllm/long-context-rag.md)
- AI for Science 中的图方法：[AI for Science 专题](../resources/ai-for-science.md)

---

## 12 参考文献 {#12}

[1] F. Scarselli et al., *The Graph Neural Network Model*, IEEE TNN, 2009.

[2] J. Bruna et al., *Spectral Networks and Locally Connected Networks on Graphs*, ICLR, 2014.

[3] M. Defferrard et al., *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering* (ChebNet), NeurIPS, 2016.

[4] T. N. Kipf & M. Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (GCN), ICLR, 2017.

[5] P. Veličković et al., *Graph Attention Networks* (GAT), ICLR, 2018.

[6] W. Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE), NeurIPS, 2017.

[7] K. Xu et al., *How Powerful are Graph Neural Networks?* (GIN), ICLR, 2019.

[8] C. Morris et al., *Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks*, AAAI, 2019.

[9] M. Zhang & P. Li, *Nested Graph Neural Networks* (GCNII 相关过平滑分析), NeurIPS, 2021.

[10] R. Ying et al., *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (PinSage), KDD, 2018.

[11] X. He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR, 2020.

[12] J. Gilmer et al., *Neural Message Passing for Quantum Chemistry* (MPNN), ICML, 2017.

[13] J. Jumper et al., *Highly Accurate Protein Structure Prediction with AlphaFold*, Nature, 2021.

[14] T. Hu et al., *Harnessing the Power of MLLMs for Transferable Graph Reasoning*（Graph Transformer 与 LLM 融合综述类），2024.

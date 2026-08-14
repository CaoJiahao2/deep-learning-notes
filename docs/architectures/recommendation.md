# 推荐系统与 CTR 模型

> 从协同过滤到深度 CTR，从多目标排序到 LLM 推荐 —— 工业界最大规模的深度学习应用。

---

## 目录

1. [概述](#1)
2. [推荐系统架构](#2)
3. [经典深度 CTR 模型](#3-ctr)
4. [多目标学习](#4)
5. [序列建模与用户行为](#5)
6. [召回与粗排](#6)
7. [LLM 与推荐系统](#7-llm)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

推荐系统是互联网内容分发的核心技术，广泛应用于电商（淘宝、Amazon）、短视频（抖音、TikTok）、新闻（头条）、广告和社交媒体。工业推荐系统是最大规模的深度学习应用之一，每天服务数十亿用户、处理万亿级样本。

推荐系统的核心问题：在海量物品（item）库中，为特定用户（user）在特定场景（context）下预测其偏好并推荐最感兴趣的物品。

与学术 benchmark 不同，工业推荐系统关注：

- **极致规模**：千亿特征、万亿样本、毫秒级延迟。
- **在线学习**：数据分布实时变化，模型需持续更新。
- **多目标平衡**：点击率、完播率、转化率、留存等需同时优化。
- **生态健康**：不能只优化短期点击，还需考虑多样性、公平性和长期用户价值。

---

## 2. 推荐系统架构

### 2.1 漏斗架构

工业推荐系统通常采用多级漏斗：

```text
召回（Matching） → 粗排（Pre-Ranking） → 精排（Ranking） → 重排（Re-ranking）
  百万级 → 千级      千级 → 百级          百级 → 几十        去重/多样性/规则
```

- **召回**：从百万/亿级物品库中快速筛选出数千候选。强调高吞吐、高召回率。
- **粗排**：对召回结果快速打分，筛选数百候选。模型轻量，延迟 < 10ms。
- **精排**：用复杂模型精确预估用户对每个物品的点击/转化概率。延迟 < 50ms。
- **重排**：考虑列表整体效果（多样性、新颖性、上下文），最终决定展示顺序。

### 2.2 特征体系

| 特征类型 | 示例 |
|---------|------|
| 用户特征 | 年龄、性别、地域、设备、历史行为统计 |
| 物品特征 | 类别、价格、标签、文本、图像、统计指标 |
| 上下文特征 | 时间、地点、场景、请求来源 |
| 交叉特征 | 用户×物品、用户×类别、物品×场景 |
| 序列特征 | 用户最近点击/购买序列、实时会话 |

### 2.3 Embedding 表示

推荐系统将稀疏 ID 特征映射为稠密 Embedding：

$$
\mathbf{e}_i = W[i]
$$

Embedding 表规模可达 TB 级（如用户数亿 × Embedding 维度 64），需专门的参数服务器（Parameter Server）或大 Embedding 引擎（如 HugeCTR、Monolith）。

---

## 3. 经典深度 CTR 模型

### 3.1 演进路线

$$
\text{LR} \rightarrow \text{FM/FFM} \rightarrow \text{Wide\&Deep} \rightarrow \text{DeepFM} \rightarrow \text{DIN} \rightarrow \text{DIEN} \rightarrow \text{SIM} \rightarrow \text{多目标/多塔}
$$

### 3.2 Wide & Deep（Google, 2016）

结合线性模型的记忆能力和深度网络的泛化能力：

- **Wide 部分**：线性模型 + 人工交叉特征，记忆特征共现模式。
- **Deep 部分**：MLP 自动学习高阶特征交叉。
- 联合训练：

$$
P(\text{click}) = \sigma(\mathbf{w}_{\text{wide}}^\top [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{\text{deep}}^\top a^{lf} + b)
$$

### 3.3 DeepFM（华为, 2017）

用 FM（Factorization Machine）替代 Wide 部分，自动学习二阶特征交叉，无需人工特征工程：

- FM 层做二阶交叉。
- DNN 层做高阶交叉。
- 共享 Embedding 层，联合训练。

FM 的二阶交互：

$$
\hat{y}_{\text{FM}} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

### 3.4 DCN / DCN-V2

Deep & Cross Network 用交叉层（Cross Layer）显式学习有限阶的特征交叉：

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 (\mathbf{w}_l^\top \mathbf{x}_l + b_l) + \mathbf{x}_l
$$

每一层增加一阶交叉，$L$ 层可学习到 $L+1$ 阶交叉，且参数高效。

### 3.5 DIN（阿里, 2018）

Deep Interest Network 根据候选物品动态激活用户历史行为中相关的部分：

- 用户兴趣是多样的，不应将所有历史行为压缩为固定向量。
- 引入注意力机制：对用户历史行为按与候选物品的相关性加权：

$$
\mathbf{v}_U(A) = f(\mathbf{v}_A) = \sum_{i=1}^{N} w(\mathbf{e}_i, \mathbf{v}_A) \mathbf{e}_i = \sum_{i=1}^{N} a(\mathbf{e}_i, \mathbf{v}_A) \mathbf{e}_i
$$

其中注意力权重 $a$ 由前馈网络计算，$\mathbf{e}_i$ 是历史物品 Embedding，$\mathbf{v}_A$ 是候选物品 Embedding。

### 3.6 DIEN（阿里, 2019）

Deep Interest Evolution Network 建模用户兴趣的演化：

- 用 GRU 对用户行为序列建模，提取兴趣演化轨迹。
- 引入辅助损失（Auxiliary Loss）用下一时刻行为监督隐状态学习。
- AUGRU（Attention-based Update Gate GRU）让候选物品相关的兴趣演化更显著。

---

## 4. 多目标学习

### 4.1 为什么需要多目标

推荐系统需要同时优化多个目标：点击率（CTR）、完播率、点赞、评论、分享、转化（CVR）、停留时长、留存等。单目标优化 CTR 可能导致标题党和低质量内容。

### 4.2 MMoE

Multi-gate Mixture-of-Experts（Google, 2018）用共享的专家网络和任务专属门控：

$$
y_k = \sum_{i=1}^{n} g_k^{(i)} f_i(x)
$$

其中 $f_i$ 是第 $i$ 个专家网络，$g_k^{(i)}$ 是第 $k$ 个任务对专家 $i$ 的门控权重。不同任务可以选择性地利用共享专家，平衡相关性与冲突。

### 4.3 PLE

Progressive Layered Extraction（腾讯, 2020）区分共享专家和任务专属专家，渐进式分离：

- 底层：所有任务共享。
- 中层：部分专家共享，部分专属。
- 顶层：完全任务专属。
- 解决 MMoE 中任务冲突严重时专家选择不稳定的问题。

### 4.4 排序融合

多目标模型输出多个预估值后，需融合为最终排序分：

- **加权求和**：$score = w_1 p_{\text{click}} + w_2 p_{\text{convert}} + \ldots$
- **帕累托优化**：寻找多目标 Pareto 最优。
- **LLM 重排**：用大模型综合多目标信号和内容理解做最终排序。

---

## 5. 序列建模与用户行为

### 5.1 长序列建模

用户终身行为序列可能长达数万条，需要高效建模：

- **SIM（阿里, 2020）**：Search-based Interest Model，先从长序列中检索与候选物品相关的子序列，再用注意力建模，支持万级长度序列。
- **ETA（End-to-end Target Attention, 2021）**：用 SimHash 做高效检索，实现端到端长序列建模。
- **TWIN（快手, 2023）**：两阶段注意力，线性复杂度处理超长序列。

### 5.2 实时行为与会话

- **实时序列特征**：用户最近几分钟的点击/搜索行为，对即时意图至关重要。
- **会话级建模**：区分单次会话内的短期意图和跨会话的长期偏好。
- **Transformer 序列模型**：用 SASRec、BERT4Rec 等自注意力序列模型做下一物品预测。

### 5.3 图神经网络推荐

用户-物品交互构成二部图，GNN 可捕获高阶协同过滤信号：

- **PinSage（Pinterest, 2018）**：基于 GraphSAGE 的 Web 规模推荐。
- **LightGCN**：简化 GCN，去除特征变换和非线性，在协同过滤上效果优异。
- **多行为图**：融合点击、购买、收藏等多种行为边。

---

## 6. 召回与粗排

### 6.1 召回方法

| 类型 | 方法 |
|------|------|
| 协同过滤 | UserCF、ItemCF、矩阵分解（SVD） |
| 双塔模型 | DSSM、YouTubeDNN、MIND |
| 图召回 | PinSage、GraphSAGE |
| 序列召回 | SASRec、BERT4Rec |
| 语义召回 | 文本/图像 Embedding + ANN |
| 规则/运营 | 热门、新品、编辑精选 |

### 6.2 双塔模型

双塔模型将用户和物品分别编码为向量，用内积/余弦相似度匹配：

$$
s(u, i) = \cos(E_u(u), E_i(i))
$$

物品向量可离线计算并建索引（FAISS/HNSW），用户向量在线计算，实现毫秒级亿级检索。关键挑战：

- 内积不能表达复杂的用户-物品交叉（信息瓶颈）。
- 需要精心设计负采样策略（全局随机负样本 + 热门物品 + 曝光未点击）。
- 知识蒸馏：将精排模型的知识蒸馏到双塔，提升召回质量。

### 6.3 粗排

粗排介于召回和精排之间，需要在低延迟下提供更精准的打分：

- 轻量交叉模型（COLD 等）。
- 知识蒸馏：用精排作为教师模型训练粗排。
- 向量内积 + 轻量特征交叉的平衡。

---

## 7. LLM 与推荐系统

### 7.1 LLM 的新能力

- **语义理解**：理解物品内容、用户查询的语义，解决传统 ID 推荐的冷启动问题。
- **零样本推荐**：LLM 可直接根据用户历史和物品描述生成推荐，无需大量交互数据。
- **对话式推荐**：用户用自然语言表达需求，LLM 多轮对话澄清意图并推荐。
- **可解释推荐**：LLM 生成自然语言推荐理由。

### 7.2 应用方式

- **LLM 作为特征编码器**：将物品标题/描述/评论编码为语义 Embedding，用于召回和排序。
- **LLM 做重排**：用 LLM 综合候选列表、用户画像和场景，输出最终排序和推荐理由。
- **LLM 增强冷启动**：对新物品，LLM 基于内容描述生成初始 Embedding 和标签。
- **Agent 推荐**：LLM Agent 主动询问用户偏好、搜索候选、比较方案。

### 7.3 挑战

- **延迟和成本**：LLM 推理成本远高于传统推荐模型，不适合精排级别的大规模打分。
- **实时性**：LLM 难以像传统模型一样分钟级更新。
- **推荐特化**：通用 LLM 对推荐场景的特有模式（位置偏差、流行度偏差）不敏感。
- 趋势是"LLM 增强传统推荐"而非完全替代：LLM 处理语义和冷启动，传统模型处理大规模实时排序。

---

## 8. 实践建议

- 入门从 GBT（LR/GBDT）和深度 CTR 模型（DeepFM/DIN）开始，理解特征工程和模型演进逻辑。
- 工业实践重视数据和特征：70% 的提升来自特征和数据质量，而非模型结构。
- 注意样本偏差：曝光未点击不等于不喜欢；需处理位置偏差、选择偏差。
- 在线 A/B 测试是最终评价标准，离线指标提升不保证线上效果。
- 关注长尾和冷启动：头部物品容易推荐，真正的挑战在长尾物品和新用户。
- LLM 目前最适合语义召回、冷启动和重排/对话场景，不适合替代精排。

---

## 9. 参考文献

[1] H.-T. Cheng, L. Koc, J. Harmsen, et al., *Wide & Deep Learning for Recommender Systems*, DLRS, 2016.

[2] H. Guo, R. Tang, Y. Ye, et al., *DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction*, IJCAI, 2017.

[3] G. Zhou, X. Zhu, C. Song, et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD, 2018.

[4] G. Zhou, N. Mou, Y. Fan, et al., *Deep Interest Evolution Network for Click-Through Rate Prediction*, AAAI, 2019.

[5] J. Ma, Z. Zhao, X. Yi, et al., *Modeling Task Relationships in Multi-Task Learning with Multi-Gate Mixture-of-Experts*, KDD, 2018.

[6] Y. Pi, X. Li, Y. Wang, et al., *Progressive Layered Extraction (PLE)*, RecSys, 2020.

[7] J. Qin, W. Zhang, X. Wu, et al., *User Behavior Retrieval for Click-Through Rate Prediction*, SIGIR, 2020.

[8] X. He, K. Deng, X. Wang, et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR, 2020.

[9] W. Kang, J. McAuley, *Self-Attentive Sequential Recommendation*, ICDM, 2018.

[10] A. Rajaraman, J. D. Ullman, *Mining of Massive Datasets*, Cambridge University Press, 2012.

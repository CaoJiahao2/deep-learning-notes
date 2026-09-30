# 自监督学习：从对比学习到掩码建模

> 不依赖人工标注，从数据自身构造监督信号——自监督学习是 BERT、GPT、CLIP、MAE、DINOv2 与机器人表征（R3M/VC-1/V-JEPA）共同的方法论地基。

---

## 目录

1. [概述：为什么要自监督](#1)
2. [代理任务时代（2015–2018）](#2)
3. [对比学习与 InfoNCE](#3)
4. [SimCLR / MoCo 工程化](#4)
5. [非对称结构：BYOL 与 SimSiam](#5)
6. [掩码建模：从 BERT 到 MAE](#6)
7. [蒸馏系：DINO / DINOv2](#7)
8. [多模态自监督：CLIP](#8)
9. [视频与机器人表征](#9)
10. [方法对比与选型](#10)
11. [实践建议](#11)
12. [相关文档](#12)
13. [参考文献](#13)

---

## 1 概述：为什么要自监督 {#1}

标注是深度学习的最大瓶颈：ImageNet 级分类标注昂贵，检测/分割标注更贵，机器人动作标注近乎不可能。自监督学习（Self-Supervised Learning, SSL）的思路是**从数据内部自动构造监督信号**（pretext task），让模型学到可迁移的通用表征，再迁移到下游任务。

它同时是三条技术线的共同源头：

- **NLP**：GPT 的下一词预测、BERT 的掩码语言建模，本质都是 SSL；
- **CV**：SimCLR/MoCo/MAE/DINOv2 的表征学习；
- **具身智能**：从视频学机器人表征（R3M、VC-1、V-JEPA），直接服务 VLA。

理解 SSL 是理解"基础模型为什么 work"的必修课。

---

## 2 代理任务时代（2015–2018） {#2}

早期 SSL 通过人工设计的代理任务学习：

- **旋转预测**：判断图像旋转角度（0/90/180/270°）——迫使模型理解物体朝向；
- **拼图（Jigsaw）**：恢复九宫格排列——学习空间结构；
- **上色（Colorization）**：灰度图预测颜色——学习语义（草地是绿的）；
- **上下文预测**（Context Prediction）：预测两个 patch 的相对位置。

局限：代理任务的"人工先验"限制了表征的上限，且对特定任务有偏。转折点是**对比学习**——不再手工设计"猜什么"，而是直接学习"什么相似、什么不同"。

---

## 3 对比学习与 InfoNCE {#3}

### 3.1 核心思想

构造正样本对（同一图像的两个增强视图）与负样本对（不同图像），在嵌入空间拉近正样本、推远负样本：

$$\text{相似}(z_i, z_j^+) \uparrow, \qquad \text{相似}(z_i, z_j^-) \downarrow$$

### 3.2 InfoNCE 损失

对比学习的信息论基础是 **InfoNCE**：最大化表示 $z$ 与上下文 $c$ 之间的互信息下界：

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\left[\log \frac{\exp(z_i \cdot z_j^+ / \tau)}{\sum_{k=0}^{N-1} \exp(z_i \cdot z_k / \tau)}\right]$$

形式上是"以正样本为正确答案的 $N$ 类分类"，温度 $\tau$ 控制难样本加权。

互信息视角：

$$I(z; c) \ge \log N - \mathcal{L}_{\text{NCE}}$$

负样本数 $N$ 越大，互信息下界越紧——这解释了对比学习对**大 batch / 大负样本队列**的执念。

---

## 4 SimCLR / MoCo 工程化 {#4}

### 4.1 SimCLR（Google, 2020）

- **强数据增强**：随机裁剪 + 颜色抖动 + 模糊的组合是正样本对的来源（增强选择本身是核心贡献）；
- **投影头**：对比损失计算在 MLP 投影后的空间 $z = g(h)$，但迁移用投影前的 $h$——"损失空间 ≠ 表征空间"的解耦；
- **大 batch**：4096，负样本即 batch 内其他样本。

### 4.2 MoCo v1–v3（Meta, 2020–2021）

MoCo 用**动量编码器 + 负样本队列**解决"负样本从哪来"：

- key 编码器参数为 query 编码器的指数滑动平均：$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$，保证队列中负样本表征的一致性；
- 队列保存上千个负样本，batch 大小解耦。

MoCo v3 把主干换成 ViT 并简化，是"对比学习 + Transformer"的标准配方。SimCLR/MoCo 系在 ImageNet 线性探测上首次超越有监督预训练。

### 4.3 对比学习的坍缩问题

若所有输入都映射到同一点，损失虽低但表征无效。对比学习靠**负样本排斥**避免坍缩——这也是它的死穴：负样本中可能混入"假负样本"（语义相同的不同实例），且负样本规模与显存强耦合。非对称结构路线（§5）直接绕开负样本。

---

## 5 非对称结构：BYOL 与 SimSiam {#5}

### 5.1 BYOL（DeepMind, 2020）

只有正样本对，靠**非对称结构**避免坍缩：

```text
x → online:  f_θ → g_θ → p_θ ─┐
                              ├→ L = ‖p_θ(z_θ) − z̄_ξ‖²  (只回传 online 分支)
x → target:  f_ξ → g_ξ ───────┘   (ξ ← 动量更新 θ, 无梯度)
```

关键组件：预测头 $p_\theta$（制造不对称）+ 动量目标网络（提供缓变目标）。实证表明缺少任一项都会坍缩。

### 5.2 SimSiam（2021）

进一步发现：**停止梯度（stop-gradient） alone 就足以防止坍缩**——损失 $\mathcal{L} = \| p_\theta(z_1) - \text{sg}(z_2) \|^2$ 无负样本、无动量也能学到好表征。理论分析（隐式优化视角）证明其动力学存在非平凡解。

意义：坍缩的避免可以来自**结构不对称**而非样本排斥，这一认知直接催生了掩码建模与蒸馏系方法的繁荣。

---

## 6 掩码建模：从 BERT 到 MAE {#6}

### 6.1 语言：BERT / GPT

- **BERT**：随机 mask 15% token，预测被遮 token（MLM）——双向编码器表征；
- **GPT**：因果语言建模，预测下一个 token——解码器表征。

语言 token 是离散、信息密集的，mask 少量即可构成充分的学习信号。

### 6.2 图像：MAE（Meta, 2022）

图像 patch 信息冗余，随机遮挡 75% 后重建：

$$\mathcal{L}_{\text{MAE}} = \| x - \text{Decoder}(f_{\text{enc}}(\tilde{x})) \|^2_{\text{masked patches}}$$

设计要点：**非对称编码-解码**——ViT 编码器只处理可见 25% patch（计算量降 4 倍），轻量解码器负责重建并在下游丢弃；重建目标是原始像素（或归一化像素）。

BEiT 用离散视觉词表（dVAE token）做 BERT 式 MLM；data2vec 统一"预测教师表征"的目标，跨模态共享配方。

### 6.3 对比 vs 掩码的哲学差异

| 维度 | 对比学习 | 掩码建模 |
|---|---|---|
| 监督信号 | 实例身份（哪个样本） | 局部内容（被遮内容） |
| 学到的特征 | 偏全局/语义 | 偏局部/纹理细节 |
| 免疫坍缩方式 | 负样本排斥 | 重建损失 |
| 下游适配 | 线性探测即强 | 需微调才发挥 |

两者后来在"更强表征"的目标下部分融合（如 iBOT = DINO + 掩码预测）。

---

## 7 蒸馏系：DINO / DINOv2 {#7}

### 7.1 DINO（Meta, 2021）

自蒸馏：学生网络与动量教师网络对同一图像的不同视图输出分布，用交叉熵拉近（而非欧氏距离）：

$$\mathcal{L} = -\sum_x \sum_t p_t(x) \log p_s(x_t), \quad p = \text{softmax}(g(f(x))/T)$$

教师输出经 centering + 锐化（温度）防坍缩。意外发现：**DINO-ViT 的自注意力图涌现出语义分割能力**——无监督下 attention 自动对准物体。

### 7.2 DINOv2（Meta, 2023）

把 SSL 做成"数据工程 + 系统工程"：

- **LVD-142M**：自动化数据管线（检索、去重、平衡）从 1.2B 图片筛出 142M 高质量图；
- **iBOT 目标 + DINO 目标联合**：掩码 token 预测 + 自蒸馏；
- **大规模稳定训练技巧**：ViT-g/14，LayerScale、SwiGLU、随机深度等。

DINOv2 表征在分类、分割、深度、检索等任务上"开箱即用"，成为 CV 领域的"通用表征底座"，也是 OpenVLA 等机器人模型的视觉编码器之一。

---

## 8 多模态自监督：CLIP {#8}

CLIP（OpenAI, 2021）用 4 亿图文对做**跨模态对比学习**：匹配的图文对为正样本，batch 内其余图文组合为负样本，双塔 ViT + Text Transformer，对称 InfoNCE：

$$\mathcal{L} = -\frac{1}{2}\left[\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}\right]$$

CLIP 的意义超越一个模型：

- 图文对齐使"开放词汇"成为可能（zero-shot 分类即 prompt 匹配）；
- 文本编码器提供了"语言条件化的视觉表征"，是文生图（Stable Diffusion 的 text encoder）、VLA（语言指令条件）的表征源头；
- 验证了"规模 + 弱监督"胜过精细标注的范式。

详见[多模态对齐与融合](../llm-mllm/multimodal-alignment.md)。

---

## 9 视频与机器人表征 {#9}

机器人的"标注"（动作轨迹）极其昂贵，从无动作标签的视频学习表征是具身智能的关键路径：

| 工作 | 机构 | 思路 | 输出 |
|---|---|---|---|
| R3M (2022) | Stanford | Ego4D 视频上时间对比 + 视频语言对齐 | 通用操作视觉表征 |
| VC-1 (2022) | Meta | 在 Ego4D + ImageNet 上做大规模 SSL（ViT-L） | 感官表征基准 |
| MVP (2023) | UCSD | MAE 式，多数据源 | 操作表征 |
| V-JEPA / V-JEPA 2 (Meta, 2024–2025) | Meta | 联合嵌入预测：在表征空间预测被遮时空块，不重建像素 | 视频世界表征 + 零样本规划 |

V-JEPA 系的意义：**在抽象表征空间做预测**（而非像素空间），学到的动力学不含纹理噪声，适合作为世界模型与机器人策略的骨干（LeCun 的 JEPA 路线）。

机器人策略实践中，视觉编码器选择已成为超参数：OpenVLA 用 DINOv2+SigLIP，V-JEPA 2、VC-1 亦可即插即用。表征迁移评估（如 Meta 的 RenDu/Bridge 类基准）显示：**SSL 视频预训练表征普遍优于 ImageNet 有监督表征**，尤其在真实家庭场景上。

---

## 10 方法对比与选型 {#10}

| 方法 | 年份 | 目标函数 | 负样本 | 计算成本 | 代表下游 |
|---|---|---|---|---|---|
| SimCLR | 2020 | InfoNCE | 大 batch 内 | 高 | 分类 |
| MoCo v3 | 2021 | InfoNCE + 动量 | 队列 | 高 | 分类/检测 |
| BYOL/SimSiam | 2020–21 | 对称 MSE + stop-grad | 无 | 中 | 分类 |
| MAE | 2022 | 掩码重建（像素） | 无 | 中（编码省 4x） | 检测/分割 |
| DINOv2 | 2023 | 自蒸馏 + iBOT | 无（教师） | 极高 | 全任务通用 |
| CLIP | 2021 | 跨模态 InfoNCE | batch 内 | 极高 | 零样本/开放词汇 |
| V-JEPA 2 | 2025 | 表征空间预测 | 无 | 高 | 视频/机器人 |

选型经验：

- **要通用底座**：直接用 DINOv2/V-JEPA 公开权重，不要重训；
- **领域图像（医疗/遥感）**：在自己数据上跑 MAE 或 DINO 自蒸馏，领域收益大；
- **小数据下游**：线性探测选对比系（表征更线性可分），微调场景掩码系更稳；
- **机器人/视频**：优先视频预训练表征（R3M/VC-1/V-JEPA）。

---

## 11 实践建议 {#11}

1. **增强即任务定义**：对比学习中裁剪尺度、颜色抖动强度直接决定学到什么不变性——增强过强会丢掉判别信息（如医疗图像的颜色就是病理信号，不能抖动）；
2. **温度与归一化**：InfoNCE 的 $\tau$ 常取 0.07–0.2，过小导致难样本主导训练不稳；
3. **投影头别省**：对比损失后接 MLP 投影头再算损失，迁移用主干输出，普遍涨点；
4. **stop-gradient 是免费的坍缩保险**：实现非对称目标时务必检查梯度是否被截断；
5. **掩码率是超参**：自然图像 MAE 用 75%，信息密集图像（医学、遥感）降到 30–50%；
6. **评估协议**：线性探测（表征质量）+ 微调（任务性能）+ k-NN 三套都要报，只看微调容易掩盖表征缺陷。

---

## 12 相关文档 {#12}

- 概率与信息论基础（互信息）：[概率论与统计深入](../fundamentals/probability-statistics.md)
- 自监督在 LLM 中的形态：[预训练](../llm-mllm/pretraining.md)、[分词与词嵌入](../llm-mllm/tokenizer-embedding.md)
- CLIP 与多模态：[多模态对齐与融合](../llm-mllm/multimodal-alignment.md)
- 自监督表征的机器人应用：[模仿学习与策略学习](../embodied-ai/policy-learning.md)、[数据集与数据采集](../embodied-ai/data.md)

---

## 13 参考文献 {#13}

[1] A. van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018. arXiv:1807.03748.

[2] T. Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML, 2020.

[3] K. He et al., *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo), CVPR, 2020.

[4] Z. Xie et al., *An Empirical Study of Training Self-Supervised Vision Transformers* (MoCo v3), ICCV, 2021.

[5] J.-B. Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning* (BYOL), NeurIPS, 2020.

[6] X. Chen & K. He, *Exploring Simple Siamese Representation Learning* (SimSiam), CVPR, 2021.

[7] J. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL, 2019.

[8] K. He et al., *Masked Autoencoders Are Scalable Vision Learners* (MAE), CVPR, 2022.

[9] H. Bao et al., *BEiT: BERT Pre-Training of Image Transformers*, ICLR, 2022.

[10] M. Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV, 2021.

[11] M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR, 2024.

[12] A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML, 2021.

[13] P. Goyal et al., *Self-Supervised Learning from Video*（综述与 VC-1 相关）, 2022–2023.

[14] S. Nair et al., *R3M: A Universal Visual Representation for Robot Manipulation*, CoRL, 2022.

[15] A. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), 2024; *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025.

[16] B. Zhou et al., *A Survey on Self-Supervised Learning for Vision*, 2024.（方法谱系综述）

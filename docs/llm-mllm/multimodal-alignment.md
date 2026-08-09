# 多模态对齐与融合架构详解

> 一句话概述：让视觉与语言在同一表示空间中"对齐"（alignment），并在模型内部"融合"（fusion）多模态信息以完成理解与生成。

---

## 目录
1. [概述](#1)
2. [对比学习与 CLIP](#2-clip)
3. [视觉编码器](#3)
4. [融合架构](#4)
5. [跨模态生成](#5)
6. [训练数据与策略](#6)
7. [评估](#7)
8. [参考文献](#8)

---

## 1. 概述

多模态学习（multimodal learning）的核心可以归结为两大目标：

- **表示对齐（alignment）**：将不同模态（如图像与文本）映射到一个共享的语义空间中，使得语义相近的跨模态样本在该空间中也彼此相近。对齐通常通过对比学习（contrastive learning）实现，模型只需"判断"两份输入是否匹配，而不必"生成"内容。
- **模态融合（fusion）**：在模型内部将多模态信息组合起来，以产生最终的输出（如生成一段描述、回答一个视觉问答）。融合要求模态之间发生深度的交互，往往伴随着生成式目标。

理解二者的区别非常关键：**对齐不需要生成**，CLIP 这类双塔模型只做对比、不生成文本；**融合需要生成**，BLIP-2、LLaVA 这类多模态大语言模型（MLLM）需要把视觉特征送入 LLM 并自回归地输出 token。本文聚焦视觉-语言（vision-language）这一最成熟的子领域，沿着"对比对齐 → 视觉编码 → 融合架构 → 跨模态生成 → 训练与评估"的主线展开。

---

## 2. 对比学习与 CLIP

### 2.1 对比学习的直觉

对比学习的核心思想极其朴素：**拉近正样本对、推远负样本对**。给定一张图像和它对应的描述（正样本对），我们希望二者在嵌入空间中尽可能接近；而这张图像与 batch 内其他图像的描述（负样本对）则应尽可能远离。

这种"匹配/不匹配"的二分类视角避免了显式生成文本的负担，因此可以用极大规模的噪声图文对进行训练，从而获得强大的通用视觉表征。

### 2.2 InfoNCE 损失推导

Oord 等人提出的 InfoNCE [2] 是 CLIP 损失的直接来源。考虑一个 batch 含 $N$ 个图文对，图像经编码器得到 $z_i^v$、文本经编码器得到 $z_i^t$（均做 L2 归一化，使内积即 cosine 相似度）。CLIP 采用对称形式的 InfoNCE：

$$
\mathcal{L}_{\text{CLIP}}=-\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(z_i^v\cdot z_i^t/\tau)}{\sum_{j=1}^{N}\exp(z_i^v\cdot z_j^t/\tau)}+\log\frac{\exp(z_i^v\cdot z_i^t/\tau)}{\sum_{j=1}^{N}\exp(z_j^v\cdot z_i^t/\tau)}\right]
$$

其中前一项是"图像到文本"方向、后一项是"文本到图像"方向的 softmax 交叉熵，$\tau$ 是可学习的温度系数。

**温度系数 $\tau$ 的作用**：$\tau$ 控制 softmax 的"锐度"。$\tau$ 越小，分布越尖锐，模型对正样本的置信度要求越高、对负样本的惩罚越强；$\tau$ 过小又会导致梯度集中在少数最难负样本上、训练不稳定。CLIP 将 $\tau$ 初始化为 $1/\sqrt{d}$ 量级并作为可学习参数联合优化。

**为何需要大 batch**：softmax 形式的 InfoNCE 的负样本来自同一 batch 内的其他样本，有效负样本数约为 $N-1$。负样本越多、越多样，对比任务越难、学到的表征越具判别力。CLIP 训练时 batch size 高达 32768，这正是其性能的关键之一——也意味着训练成本极高、复现困难。

### 2.3 CLIP 架构

CLIP [1] 采用经典的双塔结构：

- **图像塔**：ViT 或 ResNet，将图像编码为特征向量；
- **文本塔**：Transformer encoder，将文本（受最大长度限制，如 77 token）编码为特征向量；
- **投影头**：两侧各一个线性投影，将特征映射到统一的 $d$ 维空间（如 512 或 768）；
- **损失**：对称 InfoNCE 对比损失。

CLIP 最令人瞩目的能力是**零样本分类（zero-shot classification）**：把每个类别名构造成一句 prompt（如 `"a photo of a {label}."`），用 CLIP 文本塔编码所有类别的 prompt、用图像塔编码待分类图像，取 cosine 相似度最高的类别即可。这一范式让 CLIP 在不微调的情况下，于 ImageNet 上达到与有监督 ResNet 相当的水平，开启了"用自然语言做分类头"的新范式。

### 2.4 SigLIP 改进

Google 在 2023 年提出的 SigLIP [3] 指出：softmax 形式的 InfoNCE 强制要求 batch 内全部样本两两对比，这把损失与 batch size 深度耦合，导致必须超大 batch 才能训好。SigLIP 的改进是用**逐对的 sigmoid 损失**取代 softmax：

$$
\mathcal{L}=-\sum_{i=1}^{N}\log\sigma\left(z_i^v\cdot z_i^t\cdot s+b\right)
$$

其中 $s$ 是可学习的 scale、$b$ 是可学习的 bias，$\sigma$ 为 sigmoid。每个图文对独立地被判定为"匹配/不匹配"，负样本不再通过 batch 内的归一化项引入，而是通过显式构造的负样本对（同一 batch 内 $i\neq j$ 的组合视为负样本，但各自独立计算）。

| 维度 | CLIP | SigLIP |
|---|---|---|
| 损失形式 | softmax 对称 InfoNCE | 逐对 sigmoid |
| 对 batch size 的依赖 | 强（需大 batch 提供足够负样本） | 弱（小 batch 也可训好） |
| 小数据/小算力表现 | 较差，难复现 | 显著更优 |
| 大规模扩展 | 强但昂贵 | 更易扩展，已逐步成为新一代默认视觉塔 |

SigLIP 已成为许多现代 MLLM（如 PaliGemma、Idefics2/3、Cambrian 等）默认的视觉编码器，逐步取代 CLIP-ViT。

---

## 3. 视觉编码器

### 3.1 ViT 回顾

Dosovitskiy 等人在 2021 年提出的 ViT [4] 把图像当成"句子"处理：将图像切成不重叠的 patch，每个 patch 线性投影成一个 token，加上位置编码后送入标准 Transformer。其 patch 化过程可示意如下：

```text
图像 H×W×3
   │  切成 p×p 的 patch，共 (H/p)·(W/p) 个
   ▼
[patch_1] [patch_2] ... [patch_k]   (k = (H/p)·(W/p))
   │  每个 patch 经线性投影 → D 维 token
   ▼
[CLS] [tok_1] [tok_2] ... [tok_k] + 位置编码
   │
   ▼
   Transformer Encoder (L 层)
   │
   ▼
[CLS] 的输出 → 图像全局表示 (或保留全部 token 序列)
```

ViT 的一个关键特性是：**序列长度与 patch 数线性相关**。对 $224\times 224$、$p=16$ 的常见配置，序列长度为 $196$；若分辨率提到 $448\times 448$，则序列长度暴增至 $784$，自注意力的计算量随序列长度平方增长。这正是后续高分辨率 MLLM 必须解决"序列长度爆炸"的根源。

### 3.2 CLIP-ViT 作为多模态默认视觉塔

CLIP 训练完成后，其视觉塔已经与文本语义对齐，因此天然适合作为下游 MLLM 的视觉编码器——它输出的特征已经"带有语义"，连接器只需做维度适配即可送入 LLM。CLIP-ViT-L/14 几乎成了 2023 年绝大多数 MLLM（LLaVA、BLIP-2、MiniGPT-4 等）的事实标准。

### 3.3 高分辨率与 anyres 策略

真实图像分辨率千差万变，而 ViT 的位置编码和序列长度都对分辨率敏感。早期 MLLM 把图像直接 resize 到 $224$ 或 $336$，但这会丢失文档/图表中的细小文字与结构。LLaVA-NeXT [7] 等工作提出的 **anyres（any resolution）策略** 给出了一种折中：

- 将原图按长边切成 $2\times 2$ 等网格（如 $2\times 2$、$1\times 4$、$4\times 1$，按宽高比自适应选择）；
- 每个 crop 单独过一次视觉塔，再加上一个降采样的全局 crop；
- 把所有 crop 的 token 拼接送入连接器。

这样既保留了局部高分辨率细节，又复用了低分辨率预训练编码器，无需重新训练 ViT。代价是 token 数随 grid 数量倍增（如 $2\times 2$ 即 5 倍 token），给 LLM 的上下文长度带来压力。

---

## 4. 融合架构

### 4.1 三种范式概览

按"模态信息何时、何处交互"，融合架构可分为三类：

| 范式 | 交互时机 | 典型代表 | 特点 |
|---|---|---|---|
| 早期融合（early fusion） | 输入层 | PaLI [10]、原生 MLLM | 模态从一开始就交错进入同一 backbone，交互最深，但训练成本最高 |
| 晚期融合 / 双塔（late fusion） | 仅在对齐空间 | CLIP | 只在最终表示空间比较，不生成，最简单 |
| 中间融合（middle fusion） | 编码器与 LLM 之间 | BLIP-2 [5]、LLaVA [6]、Flamingo | 视觉塔与 LLM 各自独立，靠"连接器"在中间桥接，工程最灵活 |

当前主流 MLLM 几乎都采用中间融合，下面逐一详解。

### 4.2 晚期融合 / 双塔（CLIP 式）

双塔模型只允许两模态在最终的共享表示空间发生交互——通过点积或 cosine 相似度比较。它不把视觉 token 送进文本塔，也不把文本 token 送进视觉塔。这种"延迟到最后一刻才交互"的设计带来两个好处：(1) 两塔可独立前向、易缓存、易大规模检索；(2) 训练目标简单（对比损失）。代价是无法做需要细粒度跨模态交互的生成任务。

### 4.3 Q-Former（BLIP-2）

BLIP-2 [5] 提出 Q-Former 来桥接**冻结的视觉编码器**与**冻结的 LLM**。Q-Former 的核心是一组可学习的 query 向量（典型为 32 或 64 个），通过 cross-attention 从冻结的 ViT 特征中"抽取"信息，输出固定数量的 token：

```text
冻结 ViT → 视觉特征 (序列长度可变，如 196/576/...)
                              │
                              ▼  cross-attention (query→key/value)
[Q_1] [Q_2] ... [Q_K]  ──────▶  [o_1] [o_2] ... [o_K]   (K 固定，如 32)
                              │
                              ▼  线性投影到 LLM 词嵌入空间
                          固定 K 个视觉 token → 送入冻结 LLM
```

Q-Former 的关键设计是：**输出 token 数 K 固定，与输入分辨率无关**。这带来两个工程价值：

1. 解耦了视觉塔的序列长度与 LLM 的输入长度，无论用多大的图像，送入 LLM 的视觉 token 都是 $K$ 个；
2. 训练时可冻结昂贵的 ViT 和 LLM，只训练 Q-Former（参数量很小），极大降低训练成本。

代价是 $K$ 个 token 是一种"信息瓶颈"——所有视觉信息必须被压缩进 $K$ 个表示，对细粒度任务（OCR、文档理解）可能造成信息损失。

### 4.4 线性 Projector（LLaVA）

LLaVA [6] 给出了一种极简方案：用一个 MLP（LLaVA-1.5 用两层 MLP）把 ViT 输出的每个 token 投影到 LLM 的词嵌入维度，再与文本 token 拼接送入 LLM：

```text
冻结 ViT → 视觉 token 序列 (L 个 D_v 维向量)
                              │
                              ▼  MLP: D_v → D_l (LLM 嵌入维度)
                      L 个 D_l 维视觉 token
                              │
                              ▼  与文本 token 拼接
[sys] [img tokens: v_1...v_L] [user text tokens] ... → LLM 自回归生成
```

LLaVA 的设计哲学是"不要在视觉和语言之间加额外的瓶颈层"。视觉 token 数 = 视觉塔序列长度（受 anyres 的 crop 数影响），不做信息压缩。这种"简单但有效"的方案在 LLaVA-1.5、LLaVA-NeXT [7] 中持续被验证：MLP projector 配合高分辨率 anyres 多 crop，已能在多个 benchmark 上追平甚至超过使用 Q-Former 的模型。

权衡：投影方案放弃了对 token 数的显式控制，高分辨率下 token 数可能上千，对 LLM 的上下文长度和推理成本提出更高要求。

### 4.5 交错视觉 token / 原生 MLLM

Qwen-VL [8]、PaLI [10] 等模型走得更远：视觉 token 直接作为序列的一部分，原生支持图文交错输入。例如用户的输入可以是 `"这张图里 <img>图1</img> 的猫和 <img>图2</img> 的狗谁更大？"`，模型在生成时可在任意位置引用任意一张图的视觉 token。这类"原生多模态"模型通常从一开始就联合训练视觉和语言模块，而非事后拼接，因此交错能力更强，但训练成本也最高。

### 4.6 三类融合方案对比

| 方案 | 参数效率 | token 数控制 | 高分辨率支持 | 训练成本 |
|---|---|---|---|---|
| 线性 Projector（LLaVA） | 连接器参数最少 | 不可控（=ViT 序列长度×crop 数） | anyres 多 crop | 低（可冻结双塔） |
| Q-Former（BLIP-2） | 连接器参数中等 | 强（输出固定 K） | 可控但信息有瓶颈 | 中（连接器需较多训练） |
| 原生交错（Qwen-VL/PaLI） | 视觉/语言深度耦合 | 灵活 | 原生支持 | 高（通常需联合预训练） |

实践中，"线性 projector + anyres + 大 LLM" 因其简单有效，已成为开源社区 2024 年起最主流的组合。

---

## 5. 跨模态生成

前面各节聚焦"视觉到语言"的理解。完整的 MLLM 还应支持"语言到视觉/音频/视频"的生成。

### 5.1 LLM + Diffusion 两阶段

最成熟的跨模态生成范式是"LLM 出文本条件、Diffusion 模型出图像"。具体而言：

- LLM（或专门的 caption 模型）输出一段描述性文本；
- 该文本经文本编码器（如 CLIP-T 或 T5）编码为条件；
- 扩散模型（如 Stable Diffusion / SDXL / ControlNet）在该条件下迭代去噪，生成图像。

这一范式的好处是模块解耦、各司其职：LLM 负责语义规划与指令遵循，Diffusion 负责像素级合成。ControlNet 等进一步支持用边缘、深度、骨架等结构条件来约束生成。类似地，文本→音频可走 TTS pipeline，文本→视频可走 Sora 类扩散+时序建模 pipeline。

### 5.2 端到端原生多模态

另一条路线是构建"原生多模态生成模型"：所有模态（文本、图像、音频、视频）的 token 在同一 backbone 内统一建模，输入和输出都可是任意模态的混合序列。Gemini 等闭源模型、以及若干开源原生多模态工作都朝此方向努力。其优势是模态间无割裂、可端到端优化；难点是训练数据规模与模态均衡、token 化方案（图像/视频的离散或连续 token）、训练与推理成本。

### 5.3 两类范式对比

| 维度 | LLM + Diffusion 两阶段 | 端到端原生多模态 |
|---|---|---|
| 架构 | 模块解耦，文本模型 + 扩散模型 | 单一 backbone 统一处理所有模态 |
| 模态交互 | 通过文本作为中介，较间接 | 深度交互，可生成真正交错内容 |
| 训练成本 | 模块可分别训练，相对可控 | 联合训练，成本高 |
| 可控性 | 各模块可独立替换/控制 | 端到端优化，灵活性高但调试难 |
| 成熟度 | 工程成熟，主流方案 | 前沿研究方向，仍在演进 |

---

## 6. 训练数据与策略

主流 MLLM（以 LLaVA 系列为代表）通常采用两阶段训练：

### 6.1 阶段一：对齐预训练

- **目标**：让连接器（projector / Q-Former）学会把视觉特征映射到 LLM 能理解的词嵌入空间。
- **数据**：海量图文对，如图像描述数据 LAION [11]、CC3M、CC12M、SBU 等。这些数据多来自网络抓取，数量大但噪声多、描述短。
- **训练策略**：**冻结视觉塔与 LLM，只训练连接器**。使用自回归损失（next-token prediction），即给定图像预测其 caption。
- **目的**：以最低训练成本建立最基本的图文对齐能力，保留预训练 LLM 的语言能力不被破坏。

### 6.2 阶段二：指令微调

- **目标**：教会模型遵循人类指令、进行多模态对话与推理。
- **数据**：高质量多模态指令数据，如 LLaVA-Instruct（由 GPT-4 基于 COCO 图像生成多轮对话）、ShareGPT4V、LVIS-Instruct4V 等。数量远小于阶段一，但质量更高、任务更多样。
- **训练策略**：解冻 LLM（或采用 LoRA 等参数高效微调），有时也解冻视觉塔或部分解冻，使用标准自回归损失优化回答部分。
- **关键经验**：**数据质量比数量更重要**，prompt 的多样性与任务覆盖度决定模型的泛化能力。少量高质量指令数据往往胜过海量噪声数据。

两阶段流程可示意如下：

```text
┌──────────────── 阶段一：对齐预训练 ────────────────┐
│  数据：LAION / CC3M 等海量图文对（噪声大）          │
│  冻结：ViT ✅   LLM ✅   连接器 ❌（训练）          │
│  目标：图像 → caption 的自回归损失                  │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────── 阶段二：指令微调 ──────────────────┐
│  数据：LLaVA-Instruct / ShareGPT4V 等高质量指令    │
│  冻结：ViT（部分） LLM ❌（训练/LoRA） 连接器 ❌     │
│  目标：多模态对话/推理的自回归损失                  │
└────────────────────────────────────────────────────┘
                         │
                         ▼
              评估 / 部署（可选阶段三对齐微调如 RLHF）
```

部分工作（如 BLIP-2）在两阶段间还插入一个"视觉-语言表征学习"阶段（训练 Q-Former 的对比+生成任务），但 LLaVA 系列证明两阶段已足够。

---

## 7. 评估

多模态评估大致覆盖感知、推理、知识与生成几类，常见基准如下：

| 基准 | 侧重 |
|---|---|
| GQA / VQA | 基础视觉问答，考察场景图理解与常识 |
| MMBench [12] | 综合能力，含感知、推理、知识，中英双语，多选题 |
| SEED-Bench | 覆盖图像、视频、交错图文的多模态理解，多选题 |
| MMMU [13] | 大学学科级多模态理解与推理（艺术、科学、医学等），考察深度知识与推理 |
| MM-Vet | 综合能力，开放式生成，使用 LLM-as-a-Judge 评分 |
| MathVista | 多模态数学推理，结合图表、几何、函数图像等 |

**LLM-as-a-Judge 的应用与偏差**：开放式问答没有标准答案，难以用规则评分，因此 MM-Vet 等基准引入强 LLM（如 GPT-4）作为评分裁判，从"帮助性、相关性、准确性"等维度打分。这一方法极大地扩展了可评估的任务范围，但也引入已知偏差：裁判模型偏好冗长回答、偏好自身风格、对位置敏感，且对真正"超出裁判能力"的回答可能误判。使用时应配合人工抽样校准，或与多选题基准交叉验证。

---

## 8. 参考文献

[1] Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML, 2021.

[2] van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, arXiv:1807.03748, 2018.

[3] Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV, 2023.

[4] Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR, 2021.

[5] Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML, 2023.

[6] Liu et al., *Visual Instruction Tuning*, NeurIPS, 2023.

[7] Liu et al., *LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge*, arXiv, 2024.

[8] Bai et al., *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Reading, and Beyond*, arXiv:2308.09636, 2023.

[9] Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS, 2022.

[10] Chen et al., *PaLI: A Jointly-Scaled Multimodal Language-Image Model*, arXiv:2209.06794, 2022.

[11] Schuhmann et al., *LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs*, arXiv:2111.02114, 2021.

[12] Liu et al., *MMBench: Is Your Multi-modal Model an All-around Player?*, ECCV, 2024.

[13] Yue et al., *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*, CVPR, 2024.

[14] Lu et al., *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*, ICLR, 2024.

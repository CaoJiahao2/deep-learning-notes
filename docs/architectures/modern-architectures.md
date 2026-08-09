# 现代架构演进：MoE、SSM 与注意力变体

> Transformer 一统天下之后的演进：稀疏专家 (MoE) 提升参数效率、状态空间模型 (SSM/Mamba) 提供线性复杂度循环、Grouped-Query / Multi-head Latent Attention 等变体降低 KV cache 成本，以及长上下文位置编码的外推方案。

---

## 目录

1. [概述](#1)
2. [混合专家 MoE](#2-moe)
3. [状态空间模型 SSM 与 Mamba](#3-ssm-mamba)
4. [注意力变体](#4)
5. [长上下文与位置编码扩展](#5)
6. [架构选择指南](#6)
7. [参考文献](#7)

---

## 1. 概述

自 2017 年 Transformer 问世以来，它几乎一统自然语言建模的江山。然而随着模型规模与上下文长度不断扩张，原始 Transformer 的几项固有成本逐渐成为瓶颈，催生了三条并行的演进主线：

- **长上下文的二次成本**：标准 Self-Attention 对长度 $L$ 的序列需要 $O(L^2)$ 的计算与显存，无论是 4k、32k 还是百万 token，平方增长都极不友好。
- **参数效率**：密集 (dense) 模型在每次前向时激活全部参数，意味着要提升能力就必须等比例抬升 FLOPs；而稀疏激活让总参数量与单次计算量解耦。
- **推理吞吐**：自回归解码时每生成一个 token 都要读取全部历史 KV cache，显存带宽成为推理延迟的真正瓶颈，而非算力。

围绕这三条主线，社区形成了三个方向：

1. **稀疏激活的混合专家 (Mixture of Experts, MoE)**：每个 token 只路由到 top-k 个专家子网络，总参数大但单步 FLOPs 小。
2. **替代注意力循环的状态空间模型 (State Space Model, SSM)**：以 S4、Mamba 为代表，将注意力替换为线性复杂度的循环结构，训练可并行、推理常驻隐状态。
3. **注意力自身的高效变体与长上下文位置编码**：Multi-Query / Grouped-Query Attention、Multi-head Latent Attention 压缩 KV cache；RoPE 的位置插值、YaRN 等扩展上下文窗口。

本文按这三条主线依次展开，最后给出一张架构选型决策表。

---

## 2. 混合专家 MoE

### 2.1 稀疏门控思想

密集 FFN 在每个 token 上都激活全部参数。MoE 的核心想法是：**把一个大的 FFN 拆成 $N$ 个独立的"专家"FFN，每个 token 只激活其中 top-k 个**。这样模型总参数量很大（容量高），但单 token 的实际计算量只相当于一个更小的密集模型（FLOPs 低）。

最早可追溯至 Shazeer 等人 2017 年的 *Outrageously Large Neural Networks*，将 MoE 以稀疏门控的形式引入 LSTM 之上；其后 Switch Transformer、GShard 把它搬进 Transformer 的 FFN 位置。

### 2.2 门控函数 (Gating / Router)

MoE 层用一个门控网络 $G(x)$ 决定 token 应送往哪些专家。最常见的形式是 top-k 路由：

$$G(x) = \text{softmax}\big(\text{top-k}(W_g\, x)\big)$$

即先用线性矩阵 $W_g \in \mathbb{R}^{N \times d}$ 计算每个专家的 logit，取最大的 $k$ 个，**只对这 $k$ 个做 softmax，其余专家的门控权重置 0**。最终该 token 的 MoE 输出为

$$y = \sum_{i \in \text{top-k}} G_i(x)\, E_i(x)$$

其中 $E_i(\cdot)$ 是第 $i$ 个专家 FFN。未被选中的专家不参与计算，从而实现稀疏激活。

### 2.3 负载均衡辅助损失

MoE 训练有一个臭名昭著的失败模式：**路由塌缩 (router collapse)**——门控网络学会把几乎所有 token 都送到少数几个"好用"的专家，其余专家得不到训练。这会让有效专家数远小于名义专家数，参数利用率急剧下降。

Switch Transformer 引入了一个辅助损失来鼓励专家被均匀使用。设 batch 内共 $T$ 个 token、$N$ 个专家，记 $f_i$ 为送到专家 $i$ 的 token 比例，$P_i$ 为门控给专家 $i$ 的平均概率：

$$f_i = \frac{1}{T}\sum_{x} \mathbb{1}\{i \in \text{top-k}(x)\}, \quad P_i = \frac{1}{T}\sum_{x} G_i(x)$$

辅助损失定义为两者乘积的 $N$ 倍：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i\, P_i$$

当所有专家被均匀使用时 $f_i = 1/N$、$P_i = 1/N$，loss 取极小值 $\alpha$。$f_i$ 不可导，$P_i$ 可导，二者乘积把梯度回传到门控网络，从而把流量推向欠载专家。

### 2.4 代表性 MoE 模型

**Mixtral 8x7B (Mistral, 2023)**：每层 FFN 替换为 8 个专家，每个 token 选 top-2 路由。总参数约 47B，但单 token 只激活约 13B 等效密集参数，性能比肩更大的密集模型。

**DeepSeek-MoE (Dai et al., 2024)** 提出两项关键改进：

- **细粒度专家 (fine-grained experts)**：把每个标准专家进一步切小、专家总数变多，单 token 激活更多个小专家，组合更灵活、专家 specialization 更细。
- **共享专家 (shared experts)**：划出一小部分始终激活的专家，专门承载"通用知识"，避免路由专家重复学习通用模式，使路由专家更专注于领域专长。

### 2.5 训练与推理权衡

- **训练显存大**：所有专家参数都驻留显存，即便单步只激活少数，反向仍要为整层保存优化器状态。常用 expert parallelism 把不同专家分到不同 GPU，引入跨卡 all-to-all 通信。
- **推理 FLOPs 低**：相对同等总参数的密集模型，MoE 单 token 计算量显著更低；但若不精心做分桶/路由 locality，通信与调度开销会吃掉节省。
- **负载不均仍是隐患**：实际部署要监控专家利用率，必要时做 expert padding 或动态路由修正。

### 2.6 MoE 层示意与伪代码

下面给出一个 MoE 层的结构示意：

```text
                  ┌────────────────────┐
   token x  ─────►│  Router  W_g · x    │── logits ── top-k ──┐
                  └────────────────────┘                       │
                                                             ▼
        ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
        │  E_1   │  E_2   │  E_3   │  E_4   │  E_5   │  E_6   │  E_7   │  E_8   │
        └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
                              只算被选中的 k=2 个专家：
        y = G_3(x)·E_3(x)  +  G_7(x)·E_7(x)
```

下面是 top-k 路由的 PyTorch 风格伪代码：

```python
import torch
import torch.nn.functional as F

class MoELayer(torch.nn.Module):
    def __init__(self, d, n_experts, k, expert_fn):
        super().__init__()
        self.gate = torch.nn.Linear(d, n_experts, bias=False)
        self.experts = torch.nn.ModuleList([expert_fn(d) for _ in range(n_experts)])
        self.k = k

    def forward(self, x):               # x: [B, L, d]
        B, L, d = x.shape
        logits = self.gate(x)           # [B, L, n_experts]
        topk_val, topk_idx = logits.topk(self.k, dim=-1)         # 选 k 个专家
        weights = F.softmax(topk_val, dim=-1)                     # 只对 top-k softmax

        out = torch.zeros_like(x)
        for i in range(self.k):
            ei = topk_idx[..., i]                                  # 每个 token 第 i 个专家
            wi = weights[..., i].unsqueeze(-1)                    # 门控权重
            # 按 token 分发到对应专家并加权累加（实际实现用 grouped GEMM / all-to-all）
            for b in range(B):
                for t in range(L):
                    out[b, t] += wi[b, t] * self.experts[int(ei[b, t])](x[b, t])
        return out
```

> 说明：上面的双 for 只为讲清楚语义，真实实现用 `torch.scatter` / grouped GEMM / Megatron 的 expert-parallel all-to-all。

---

## 3. 状态空间模型 SSM 与 Mamba

### 3.1 连续状态空间模型

状态空间模型源自控制论中的线性时不变 (LTI) 系统：输入 $x(t)$ 经隐状态 $h(t)$ 映射到输出 $y(t)$：

$$h'(t) = A\, h(t) + B\, x(t), \quad y(t) = C\, h(t)$$

其中 $A \in \mathbb{R}^{N \times N}$、$B \in \mathbb{R}^{N \times 1}$、$C \in \mathbb{R}^{1 \times N}$（单输入单输出形式；多维输入输出推广同理）。通过零阶保持 (Zero-Order Hold, ZOH) 离散化，给定步长 $\Delta$，得到离散参数 $\bar{A}, \bar{B}, \bar{C}$ 后即可在序列上 rollout：

$$h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t, \quad y_t = \bar{C}\, h_t$$

这是一个线性循环，因此既可像 RNN 那样逐步递推（推理 $O(L)$），也可借助卷积/扫描并行训练。

### 3.2 S4：结构化状态空间

直接学一个 $N \times N$ 矩阵 $A$ 既难收敛又不利长程建模。**S4 (Gu et al., 2021)** 的关键贡献是用 HiPPO 矩阵初始化 $A$，使其隐状态对近端历史有指数衰减的记忆、对远端历史有压缩表示，从而获得远超普通 RNN 的长序列建模能力。S4 还通过结构化参数化（对角加低秩）把推理时的卷积运算降到 $O(L)$。

### 3.3 Mamba 与选择性 SSM

S4 仍是 LTI 系统：$A, B, C, \Delta$ 与输入无关，对所有 token 一视同仁。这让它在长序列"信息检索"上力不从心——它无法根据内容选择性地记住或遗忘。

**Mamba (Gu & Dao, 2023)** 的核心突破是 **选择性 SSM (Selective SSM)**：让 $B$、$C$ 与离散化步长 $\Delta$ 依赖于输入 $x_t$：

$$B_t = B(x_t), \quad C_t = C(x_t), \quad \Delta_t = \Delta(x_t)$$

$\Delta_t$ 控制离散化"步长"，可理解为一种门控：$\Delta_t$ 大时新输入强烈更新隐状态（"记住"），$\Delta_t$ 小时几乎忽略输入（"遗忘"）。这样把 LTI 打破成内容相关、时变的循环，类似门控 RNN，但保持线性复杂度。

### 3.4 硬件感知扫描

选择性参数意味着不能再像 S4 那样用固定卷积核，必须做时变循环。直接实现 $O(L)$ 串行扫描在 GPU 上极慢。Mamba 采用 **hardware-aware parallel scan**：把递推写成结合律的扫描算子，用前缀和式的并行算法在 GPU 上高效执行，避免显式物化 $N \times N$ 的状态转移矩阵、保持显存与算力高效。

### 3.5 Mamba-2

**Mamba-2 (Dao & Gu, 2024)** 进一步发现选择性 SSM 与线性注意力之间存在 **状态空间对偶性 (State Space Duality, SSD)**：把 SSM 的结构化矩阵与注意力的核矩阵对齐，使 Mamba 能像注意力一样用矩阵形式并行计算，速度更快、更易扩展，长序列上吞吐显著提升。

### 3.6 与 Transformer 对比

| 维度 | Transformer (Attention) | SSM / Mamba |
|---|---|---|
| 序列复杂度 | $O(L^2)$ | $O(L)$ |
| 训练并行 | 高（矩阵乘） | 中（parallel scan） |
| 长上下文 | 受 KV cache 显存限制 | 隐状态固定，天然长 |
| 精确检索/复制 | 强 | 弱（LTI 残余 + 线性瓶颈） |
| 推理状态 | KV cache 随 $L$ 线性增长 | 固定大小隐状态 |
| 推理延迟 | 带宽受限 | 算力受限 |

需要特别指出 SSM 的短板：**精确检索能力弱**。Arora 等人 (*The Binary Task*, 2024) 的 "Zoology" 系列研究表明，纯 SSM/Mamba 在需要逐字复制远距离 token 的任务上明显弱于注意力，因为线性状态空间难以在任意长距离上做精确 lookup。这也是混合架构兴起的直接原因。

### 3.7 混合架构 Hybrid SSM + Attention

为兼得线性复杂度的长上下文与注意力的检索能力，业界推出多款 **混合架构**：

- **Jamba (AI21, 2024)**：在 Transformer 块中交错放置 Mamba 层与注意力层，并辅以 MoE，单层显存与吞吐显著优于纯注意力，又能做精确检索。
- **Mamba-2 + Attention 交错**：每若干层 Mamba-2 插一层 attention，把"线性扫描记忆"与"全局检索"分工。

经验上每隔 4–8 层插 1 层 attention，即可在多数长上下文基准上追平纯注意力，同时显存与吞吐大幅改善。

---

## 4. 注意力变体

即便保留注意力，标准 Multi-Head Attention 在解码时的 KV cache 仍是显存带宽瓶颈：每个 head 都要为历史 token 缓存 K、V。围绕"如何更小地存 KV"，社区给出一条从极端激进到优雅的演进链。

### 4.1 Multi-Query Attention (MQA)

Shazeer 2019 提出 **Multi-Query Attention**：所有 query head 共享同一组、仅 1 个 head 的 K/V。这样 KV cache 最小，但 query 仍有多 head 的多样性。代价是质量通常略掉点，尤其大模型上更明显。

### 4.2 Grouped-Query Attention (GQA)

**GQA (Ainslie et al., 2023)** 取折中：K/V head 数 $G$ 介于 1 (MQA) 与 $H$ (MHA，标准多头) 之间，每个 K/V head 被一组 query head 共享。

设 query head 数 $H$、K/V head 数 $G$、head 维度 $d$、序列长度 $L$，则推理时 KV cache 大小对比：

| 注意力形式 | KV cache 大小 |
|---|---|
| 密集 MHA | $\sim H \cdot L \cdot d$ |
| GQA | $\sim G \cdot L \cdot d$ |
| MQA | $\sim L \cdot d$ |

GQA 在 $G=1$ 退化为 MQA、在 $G=H$ 退化为 MHA，因而可在质量与显存间任意插值。**Llama 2 (部分大尺寸) / Llama 3 全系采用 GQA**，已成为现代 LLM 的事实标准配置之一。

### 4.3 Multi-head Latent Attention (MLA)

DeepSeek-V2 (2024) 提出 **Multi-head Latent Attention**：把 KV 压缩成一个低秩的 **latent 向量**，推理时只缓存这个 latent，再上投影恢复出各 head 的 K/V。

设隐状态维度 $d$、latent 维度 $d_c \ll d \cdot H$，则每 token：

$$c = W_{\text{down}}\, x \in \mathbb{R}^{d_c}, \quad K, V = \text{reshape}(W_{\text{up}}\, c)$$

推理 KV cache 从 $H \cdot L \cdot d$ 降到 $d_c \cdot L$，量级接近 MQA 却几乎不掉点。一个细节难点是 RoPE：latent 上投影后再做旋转位置编码会破坏低秩压缩，故 MLA 采用 **解耦 RoPE**——把位置相关的部分单独做一小段不可压缩的 K，与 latent 恢复出的内容 K 拼接使用。MLA 是 DeepSeek-V2/V3 实现超长上下文 + 低成本推理的关键一环。

### 4.4 线性注意力近似

另一条思路是 **用核技巧把 softmax 注意力近似为线性**，从而把复杂度从 $O(L^2)$ 降到 $O(L)$。一般形式（Katharopoulos et al., 2020）：

$$\text{Att}(q, k, v) = \frac{\phi(q)\,\big(\phi(k)^\top v\big)}{\phi(q)\,\sum \phi(k)}$$

由于 $\phi(k)^\top v$ 可先算再与 $\phi(q)$ 相乘，可分解为类似 RNN 的递推：

$$S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \quad y_t = \frac{\phi(q_t)\, S_t}{\phi(q_t)\, \sum_{i\le t}\phi(k_i)}$$

从而推理时 $O(L)$、训练时仍可并行。**Performer (Choromanski et al., 2020)** 用随机特征 (random features) 近似 softmax，给出无偏估计。

但线性近似普遍存在 **精度损失**：去掉指数核后注意力分布更"平"，对精确检索任务退化明显，且随机特征方差偏大。因此线性注意力在工业大模型中仍不及 GQA/MLA 主流，更多用于超长上下文的探索性工作。

---

## 5. 长上下文与位置编码扩展

### 5.1 RoPE 回顾

现代 LLM 普遍采用 **旋转位置编码 (Rotary Position Embedding, RoPE, Su et al., 2021)**：把 query、key 视作二维平面的复数，按位置 $m$ 旋转角度 $m\theta$：

$$\text{RoPE}(x, m) = x\, e^{i m \theta}$$

对一对位置 $m, n$ 的 query/key，相对相位为 $(m-n)\theta$，因此 RoPE 天然编码 **相对位置**，且对训练时未见过的更长距离外推能力有限。

### 5.2 外推问题

模型在 $L_{\text{train}}$（如 4k）上训练，推理用到 $L_{\text{test}} > L_{\text{train}}$（如 32k、128k）时，超出训练范围的相对相位会让注意力分布失常、质量骤降。围绕"如何把 RoPE 外推到更长上下文"涌现一批方法。

### 5.3 位置插值 (Position Interpolation)

**Position Interpolation, Chen et al., 2023**：与其外推到未见角度，不如 **把目标位置压缩到训练范围内**——把旋转角度缩放 $s = L_{\text{test}} / L_{\text{train}}$：

$$\theta \to \theta / s$$

即令模型"以为"仍在训练长度内。该法通常需要少量长文本微调来恢复质量，但稳定性远好于直接外推。

### 5.4 NTK-aware / Dynamic NTK

NTS 系列方法调整 RoPE 基频 $\theta_{\text{base}}$，使低频部分更平滑、高频部分基本不变。直觉上：高频对应细粒度局部位置、应保持原行为；低频对应长程、应平滑外推。

- **NTK-aware interpolation**：把基频按缩放因子放大，等价于对频率做非线性重映射。
- **Dynamic NTK**：在生成过程中随当前长度动态调整缩放，越接近外推区越平滑过渡，免微调即可小幅扩窗。

### 5.5 YaRN

**YaRN (Peng et al., 2023)** 在 NTK-aware 基础上更进一步：对不同频段做 **分段插值**——高频段不插值、中频段做 NTK、低频段做常数插值，并配合一个 attention scaling 因子调节温度。YaRN 可用极少微调（甚至仅 400 步）把上下文从 4k 扩到 128k，是 Llama 系列社区长上下文方案的事实标准之一。

### 5.6 方法对比

| 方法 | 是否需微调 | 支持上下文 | 质量损失 |
|---|---|---|---|
| 直接外推 | 否 | 略超训练长度 | 大 |
| Position Interpolation | 是（少量） | 数倍 | 中 |
| NTK-aware / Dynamic NTK | 否 | 2–4 倍 | 小 |
| YaRN | 极少 | 8–32 倍 | 小 |

### 5.7 注意力 sink 与滑动窗口

除了改位置编码，还有一类 **结构性扩窗** 方案：

- **Attention sink (StreamingLLM, Xiao et al., 2023)**：发现稳定长上下文的关键是保留前几个 token 作为"注意力沉降槽"，自回归推理时只需保留 sink + 滑动窗口即可近乎无限外推，虽不能精确回忆中段内容，但整体生成质量稳定。
- **Sliding Window Attention (Mistral)**：每层只看最近 $W$ 个 token，配合层数堆叠让有效感受野随层增长，显存与计算恒定；遇到需要远距离精检索的任务时再叠加少量全局 attention。

---

## 6. 架构选择指南

下面给出一张决策表，帮助在不同场景间选型。维度包括训练成本、推理吞吐、长上下文、检索需求、部署目标。

| 场景 / 约束 | 推荐架构 | 理由 |
|---|---|---|
| 通用大模型、追求 SOTA 质量、算力充裕 | 密集 Transformer + GQA | 检索能力强、生态最成熟、工程栈完备 |
| 参数预算大但推理 FLOPs 想省 | MoE（Mixtral / DeepSeek-MoE 式） | 总容量高、单 token 算力低，性价比高 |
| 极长上下文、显存敏感、可容忍检索略降 | Mamba / Mamba-2 | 线性复杂度、固定隐状态、长序列吞吐优 |
| 既要长上下文又要精确检索 | Hybrid SSM + Attention（如 Jamba） | Mamba 承载长程、注意力负责精检索 |
| 推理显存带宽极紧（边缘 / 高并发） | GQA 或 MLA | KV cache 最小，带宽友好；MLA 在超长下更优 |
| 需要超长上下文但不想重训 | RoPE 外推（YaRN / PI 微调） | 在已有模型上低代价扩窗 |

简明推荐：

- 若从头训大模型且追求通用能力，**密集 Transformer + GQA** 仍是最稳妥的默认选择；
- 若希望在同等推理算力下把容量做厚，**MoE** 是性价比最高的路径；
- 若业务以超长上下文（> 100k）为主且对逐字检索不苛刻，**Mamba-2 / Hybrid** 值得优先评估；
- 若要在已有 RoPE 模型上扩窗，**YaRN** 是当前最稳的轻量方案。

最终选型应回到具体业务的延迟、显存、质量三角，而非追逐单一论文的 SOTA。

---

## 7. 参考文献

[1] Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR, 2017.
[2] Fedus et al., *Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR, 2021.
[3] Jiang et al., *Mixtral of Experts*, arXiv:2401.04088, 2023.
[4] Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, ACL, 2024.
[5] Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*, ICLR, 2022.
[6] Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752, 2023.
[7] Dao and Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)*, ICML, 2024.
[8] Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*, arXiv:2403.19887, 2024.
[9] Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need (MQA)*, arXiv:1911.02150, 2019.
[10] Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP, 2023.
[11] DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)*, arXiv:2405.04434, 2024.
[12] Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML, 2020.
[13] Choromanski et al., *Rethinking Attention with Performers (FAVOR+)*, ICLR, 2021.
[14] Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)*, Neurocomputing, 2021.
[15] Chen et al., *Extending Context Window of Large Language Models via Position Interpolation*, arXiv:2306.15595, 2023.
[16] Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, ICLR, 2024.
[17] Arora et al., *The Biology of the Binary Task: SSMs and the Zipper Theorem (Zoology)*, arXiv:2402.19443, 2024.
[18] Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR, 2024.

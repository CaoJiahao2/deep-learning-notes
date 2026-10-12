# Genie：从无标注视频学习生成式交互环境

> Genie 从 20 万小时游戏视频中学习潜在动作空间和下一帧动态，无需人工动作或文本标注，就能从图像生成可交互的基础世界模型。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | Genie: Generative Interactive Environments (2024) |
| 机构 | Google DeepMind / University of Alberta / UC Berkeley |
| 关键词 | latent action model、interactive video world model、MaskGIT、ST-Transformer、foundation world model |
| 定位 | 生成式交互世界模型的奠基论文 |

---

## 1 背景与问题

普通视频生成模型预测“接下来画面可能是什么”，但不提供可控接口。一个真正的交互环境需要：

- 用户动作能影响未来；
- 世界状态长期一致；
- 能从 prompt 生成新环境；
- 不依赖昂贵动作标注。

Genie 的关键问题：能否只从无标注视频中学到动作和动态？


---

## 2 核心贡献

1. 提出 generative interactive environment 概念；
2. 使用超过 200,000 小时公开互联网游戏视频训练；
3. 构造三个组件：video tokenizer、latent action model、dynamics model；
4. 全部组件使用 spatiotemporal Transformer；
5. 无动作、无文本标注学习可控交互；
6. 模型规模最高约 11B，并对 40M 到 2.7B 做 scaling 分析；
7. 可将任意图像变成可交互虚拟世界，是后续 Genie 2 / Genie 3 的基础。


---

## 3 核心原理与方法详解

Genie 的自监督循环为：

**1. Video tokenizer**

把帧 $x_t$ 压成离散 token：

$$
z_t = E(x_t).
$$

**2. Latent action model**

从相邻帧推断潜在动作：

$$
\hat{a}_t \sim p_\alpha(\hat{a}_t\mid z_{\le t}).
$$

该模块通过信息瓶颈只保留对预测下一帧必要的控制信息。

**3. Dynamics model**

给定历史 token 和 latent action 预测下一帧 token：

$$
p_\theta(z_{t+1}\mid z_{\le t}, \hat{a}_t).
$$

推理时用户选择离散 latent action，模型逐帧生成环境：

```text
prompt image / text
   ↓
initial world tokens
   ↓ ← user latent action
next-frame dynamics
   ↓
decode / next token
```

模型使用 MaskGIT 式并行解码，而不是逐 pixel 生成。


---

## 4 数据与训练

- 数据为公开互联网游戏视频，无动作标签；
- 论文清洗并筛选以保持环境可控性；
- Latent action model 采用 causal attention，只看过去；
- Dynamics model 使用 spatiotemporal attention 捕捉空间结构和时间记忆；
- 模型规模覆盖 40M、300M、700M、1.4B、2.7B，完整系统约 11B；
- 评测包括可控性、环境一致性、prompt 泛化和 scaling 曲线；
- 用户通过离散 latent action 控制，而不是自然语言动作。


---

## 5 实验、结果与解读

Genie 的实验显示：

- 能生成角色可控、环境一致、可长期交互的 2D 平台游戏世界；
- 对未见 hand-drawn sketch / generated image prompt 有零样本生成能力；
- latent action space 无监督涌现，对应左移、右移、跳跃等控制语义；
- 模型规模和 batch size 增大会改善视频质量与交互一致性；
- 但时长、分辨率和物理一致性仍明显受限。


---

## 6 精读要点与实践建议

- **没有动作标注也能学控制接口**，这是视频世界模型成为环境的关键；
- **latent action 是信息瓶颈**：只保留会影响未来的用户输入，过滤无关噪声；
- **交互一致性比单帧 FVD 更重要**；
- **基础模型化路径**：一个模型生成无数环境，而不是为每个游戏手工建模；
- **实践启示**：视频预训练可为机器人提供动态先验，但还需要与真实动作对齐；
- **与 V-JEPA 的区别**：Genie 生成像素 / token，V-JEPA 在抽象表征中预测。


---

## 7 局限与后续影响

- 主要限于 2D 平台游戏视频，不能直接生成复杂 3D 物理世界；
- 潜在动作不保证与机器人动作空间对应；
- 长时记忆和一致性有限；
- 分辨率与实时性不足；
- 后续 Genie 2 / Genie 3 分别扩展到 3D 世界、实时交互与更长一致性。


---

## 相关阅读

- [Genie 3 世界模型精读](./genie-3.md)
- [Cosmos 世界模型精读](./cosmos.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. Bruce et al. *Genie: Generative Interactive Environments*, 2024. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
2. Chang et al. *MaskGIT: Masked Generative Image Transformer*, CVPR 2022.
3. Ha & Schmidhuber. *World Models*, 2018.


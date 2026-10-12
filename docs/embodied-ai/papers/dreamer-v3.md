# DreamerV3：单一配置跨 150+ 任务的通用世界模型 RL

> DreamerV3 通过 RSSM 想象未来并训练 actor-critic，以一套固定超参数跨域解决 Atari、ProcGen、Crafter、Minecraft 等任务。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | DreamerV3: Mastering Diverse Domains through World Models (2023) |
| 机构 | University of Geneva / DeepMind / University of Toronto |
| 关键词 | model-based RL、RSSM、imagination training、symlog、general algorithm |
| 定位 | 世界模型用于决策的经典强基线 |

---

## 1 背景与问题

强化学习的通用性长期受两件事限制：

1. 不同环境需要不同 reward scale、网络结构、探索与超参数；
2. model-free 方法样本效率低，难以从像素学习长程任务。

DreamerV3 的目标是做一个 **fixed configuration** 的通用算法：不逐任务调参，跨过 150 个以上任务稳定学习。


---

## 2 核心贡献

1. 基于循环状态空间模型 RSSM 学习隐空间世界模型；
2. 在想象 rollout 中训练 actor 和 critic，而非只在真实环境采样；
3. 使用 symlog 编码、二值化/两 hot 回归、free bits、KL balancing 等稳定化技巧；
4. 在 Atari 100k、Atari 10M、ProcGen、DMLab、Crafter、Minecraft 等基准上与专用 SOTA 竞争；
5. 首个不使用人类数据与课程、从零开始收集 Minecraft 钻石的算法；
6. 展示世界模型 RL 可以成为一种通用问题求解器。


---

## 3 核心原理与方法详解

Dreamer 学习三部分：

**1. RSSM 世界模型**

将历史观测与动作映射到随机与确定性 latent：

$$
h_t=f(h_{t-1},z_{t-1},a_{t-1}),
$$

$$
z_t\sim p_\theta(z\mid h_t),
\quad
\hat{z}_{t+1}\sim p_\theta(\hat{z}\mid h_t).
$$

同时训练重构/ reward / continue 预测头。

**2. Actor**

在想象轨迹上最大化回报：

$$
\max_\phi
\mathbb{E}_{q_\theta,\pi_\phi}
\sum_{\tau=t}^{t+H}
\gamma^{\tau-t} V_\psi(\hat{s}_\tau).
$$

**3. Critic**

学习 $V_\psi(\hat{s}_\tau)$，并可用 distributional two-hot 回归表示回报分布。

训练循环：

```text
真实环境采集少量 transition
        ↓
更新 RSSM / decoder / reward / continue
        ↓
从当前 latent 想象 H 步
        ↓
在想象中更新 actor 与 critic
        ↓
用 actor 继续真实交互
```


---

## 4 数据与训练

- 固定超参数，不按游戏逐任务调参；
- 对 reward、continue、value 使用稳定化变换；
- KL balancing 与 free bits 防止 latent collapse；
- 使用 Recurrent State Space Model 处理部分可观测性；
- 支持 pixels 与 low-dimensional state 输入；
- 论文给出大规模消融，说明每个稳定化设计的贡献；
- Minecraft 钻石任务从零开始，无人类数据、无课程。


---

## 5 实验、结果与解读

关键结果：

- 在 150+ 多样任务上超过许多专用算法；
- Atari 100k、Atari、ProcGen、DMLab、Crafter、Visual Control、Proprio Control 上均有强表现；
- Minecraft 中首次从零收集钻石，说明长程探索与规划能力；
- 相比 DreamerV2，稳定性和跨域泛化显著提升；
- 单一配置降低 RL 应用门槛。


---

## 6 精读要点与实践建议

- **世界模型的价值要以决策回报衡量**，不是只看预测误差；
- **稳定化比新结构更重要**：数值尺度、KL 约束和回报分布决定能否跨域；
- **想象训练省真实样本**，适合仿真昂贵或真机危险场景；
- **RSSM 是连接像素与控制的标准中间层**；
- **实践建议**：机器人上可先训练 RSSM 表征，再冻结或微调用于 VLA / 规划；
- **与视频世界模型对比**：Dreamer 在 latent 中推演，Genie / Cosmos 在像素或 token 空间生成。


---

## 7 局限与后续影响

- 仍依赖环境 reward 与大量交互；
- 想象 rollout 长度与模型误差限制长程规划；
- 对开放真实世界视觉泛化不如互联网视频模型；
- 与语言指令和大规模语义知识结合不足；
- 机器人真机上不能直接无限制试错。


---

## 相关阅读

- [World Models 论文精读](./world-models-2018.md)
- [Genie 论文精读](./genie.md)
- [强化学习与对齐](../../training/reinforcement-learning.md)

---

## 参考文献

1. Hafner, Pasukonis, Ba, Lillicrap. *DreamerV3: Mastering Diverse Domains through World Models*, 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
2. Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination*, ICLR 2020.
3. Ha & Schmidhuber. *World Models*, 2018.


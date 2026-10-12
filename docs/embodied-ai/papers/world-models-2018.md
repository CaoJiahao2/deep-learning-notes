# World Models（2018）：在梦中训练智能体

> Ha 与 Schmidhuber 的 World Models 把感知、记忆和控制拆成 V、M、R 三模块，并提出让智能体在自己的内部世界模型生成的“梦”中训练。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | World Models (2018) |
| 机构 | Brain Corporation / IDSIA / Google Brain |
| 关键词 | VAE、MDN-RNN、controller、dream training、world model |
| 定位 | 现代世界模型与 model-based RL 的概念源头 |

---

## 1 背景与问题

强化学习通常把环境当作黑盒，策略同时学习感知、记忆与控制，导致样本效率低。人类并非如此：我们拥有内部世界模型，可以在行动前预测后果。

World Models 的核心问题是：如果先学一个环境模型，再让控制器在这个模型内部训练，是否能学到更紧凑、更样本高效的策略？


---

## 2 核心贡献

1. 将智能体拆成视觉编码器 V、记忆模型 M 和控制器 R；
2. 用 VAE 学习视觉压缩表示；
3. 用 MDN-RNN 预测下一隐状态，形成记忆和动态模型；
4. 用线性/小型 controller 根据 $z_t$ 与 $h_t$ 输出动作；
5. 提出 dream training：在 world model rollout 中训练 controller，再迁移回真实环境；
6. 在 Car Racing 与 VizDoom 中验证内部模型足以支持策略学习。


---

## 3 核心原理与方法详解

三模块形式：

**Vision（V）**：VAE encoder 把帧 $x_t$ 压成隐变量 $z_t$：

$$
z_t \sim q_V(z\mid x_t).
$$

**Memory（M）**：MDN-RNN 预测下一隐状态分布：

$$
p_M(z_{t+1}\mid z_t,a_t,h_t).
$$

**Controller（R）**：小型线性策略：

$$
a_t = W_z z_t + W_h h_t + b.
$$

训练流程：

```text
随机策略交互
    ↓
训练 VAE 压缩视觉
    ↓
训练 RNN 预测动态
    ↓
在 RNN 生成 rollout 中优化 controller
    ↓
回到真实 / 游戏环境执行
```

 dreaming 时模型可从 $z_t,h_t$ 采样未来，无需真实渲染。


---

## 4 数据与训练

- V 采用 Variational Autoencoder，将高维像素压缩成低维 latent；
- M 使用 Mixture Density RNN 处理环境的随机与多峰转移；
- Controller 参数极少，可用进化策略或 CMA-ES 优化；
- 论文还通过隐状态噪声调节 dream 的随机性，避免策略过度利用模型偏差；
- 实验环境包括 Car Racing 和 VizDoom take-cover 任务；
- 该工作影响了 PlaNet、Dreamer、MuZero、Genie、Cosmos 等后续模型。


---

## 5 实验、结果与解读

实验显示：

- 在 Car Racing 中，只使用 VAE latent 与 RNN 记忆即可达到竞争性能；
- 在 VizDoom 中，controller 可主要依赖 RNN 记忆而非当前帧；
- dream training 能学到有效策略，但如果内部模型有偏差，策略回到真实环境会退化；
- 视觉压缩显著降低控制问题维度；
- 动态模型质量比像素重构质量更影响策略。


---

## 6 精读要点与实践建议

- **世界模型的核心不是生成漂亮画面，而是提供决策可用的状态转移预测**；
- **V / M / R 拆分**对应现代 VLA 中的感知编码、动态预测与动作头；
- **dream training 的风险**：模型误差会被策略利用；
- **实践建议**：先评估 latent dynamics 的预测与规划价值，再看视频 FVD；
- **与现代对比**：Dreamer 把 dream training 做成稳定 actor-critic，视频世界模型把它扩展到互联网视频。


---

## 7 局限与后续影响

- 低容量 VAE / RNN 只能处理简单 2D 与局部 3D 游戏；
- 世界模型没有动作 grounding 时可能学到相关性而非因果；
- 长时 rollout 误差累积；
- 对高维真实机器人任务不可直接使用；
- 后续 Dreamer 系列通过 RSSM、actor-critic 与稳定化技术解决部分问题。


---

## 相关阅读

- [DreamerV3 论文精读](./dreamer-v3.md)
- [Genie 论文精读](./genie.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. Ha & Schmidhuber. *World Models*, 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
2. Hafner et al. *Learning Latent Dynamics for Planning from Pixels*, ICML 2019.
3. Hafner et al. *DreamerV3: Mastering Diverse Domains through World Models*, 2023.


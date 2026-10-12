# RDT-1B：面向双臂操作的扩散 Transformer 基础模型

> RDT-1B 用 1.2B 参数 Diffusion Transformer 和物理可解释统一动作空间，专门解决双臂协调中的多峰动作分布与数据稀缺问题。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation (ICLR 2025) |
| 机构 | Tsinghua University / Tsinghua-Bosch Joint ML Center 等 |
| 关键词 | diffusion policy、bimanual manipulation、unified action space、cross-robot pretraining |
| 定位 | 开源连续生成式 VLA 与双臂操作基础模型代表作 |

---

## 1 背景与问题

双臂任务比单臂困难得多：

- 左右手可能有多条合理轨迹，形成多峰分布；
- 双臂自由度更高，动作间强耦合；
- 双臂遥操作昂贵，任务数据稀缺；
- 不同机器人动作空间差异更大，直接混合会破坏物理含义。

RDT-1B 的答案是用扩散模型表达多峰动作分布，并设计统一但保留物理量的动作空间。


---

## 2 核心贡献

1. 提出 Robotics Diffusion Transformer（RDT），用于可扩展的双臂操作基础模型；
2. 用 Diffusion 建模多峰连续动作块分布；
3. 提出 Physically Interpretable Unified Action Space（USV）；
4. 在 46 个数据集、1M+ 轨迹上预训练，规模至 1.2B；
5. 自建 6K+ 双臂多任务数据集并微调；
6. 展示零样本泛化、语言遵循、1–5 shot 新技能学习和复杂双臂任务能力。


---

## 3 核心原理与方法详解

RDT 的去噪过程学习：

$$
\epsilon_\theta(\tilde{a}_{t:t+H}, k\mid o_t, s_t, \ell)
\approx \epsilon,
$$

其中 $k$ 是 diffusion timestep，$\tilde{a}$ 是加噪动作块，$\epsilon$ 是噪声。推理时从噪声迭代去噪得到动作块。

模型结构：

```text
RGB 图像 → SigLIP
语言指令 → T5-XXL
本体状态 / 噪声动作 → 时间嵌入 + 输入投影
        ↓
Robotics Diffusion Transformer
        ↓
去噪后的连续动作块
```

**统一动作空间（USV）** 是 128 维向量，按物理含义分槽：左臂关节、右臂关节、夹爪、底盘、升降等。某个机器人没有的维度填充特殊值，而不是把所有动作压成无语义 token。

RDT 还使用频率编码增强高维机器人状态与非线性信号表达。


---

## 4 数据与训练

- 预训练数据：46 个多机器人数据集，超过 1M trajectories；
- 视觉与语言编码器冻结，节省显存并保留预训练表征；
- 先跨机器人预训练，再在目标双臂机器人上微调；
- 自建数据包含递水、折叠短裤、倒水到指定刻度、遛机器狗等任务；
- 动作使用 chunk 形式，适合连续控制与平滑执行；
- 模型规模 1.2B，是发布时较大的 diffusion-based manipulation foundation model。


---

## 5 实验、结果与解读

论文报告的核心结果：

- 目标任务上 RDT 达到约 68.2% 成功率，而 from-scratch 仅约 1.7%，OpenVLA 与 Octo 也明显更低；
- 预训练带来约 33.4% 绝对提升；
- 1-shot 与 5-shot 新任务学习显著优于基线；
- 对未见杯子和房间有零样本泛化；
- 能执行语言条件任务与双手协作任务；
- 消融说明统一动作空间与扩散头是多机器人迁移的关键。


---

## 6 精读要点与实践建议

- **动作表示决定可迁移性**：物理含义分槽比零填充到黑盒 token 更容易跨机器人共享；
- **Diffusion 适合双臂多峰分布**：同一初始状态可有多条左右手轨迹；
- **冻结视觉/语言编码器**是性能与算力的有效折中；
- **实践建议**：双臂数据少时，应尽量利用单臂与异构机器人预训练；
- **评测重点**：成功率之外要看 few-shot 数据效率与泛化对象/场景。


---

## 7 局限与后续影响

- Diffusion 多步去噪推理比 Flow Matching 慢；
- 1.2B 模型对边缘部署仍有压力；
- 统一动作空间需要人工设计槽位与物理量；
- 预训练数据质量与目标双臂任务差异会影响迁移；
- 长程任务和世界状态推理仍需额外规划。


---

## 相关阅读

- [π0 论文精读](./pi-0.md)
- [OpenVLA 论文精读](./openvla.md)
- [操作学习](../manipulation.md)

---

## 参考文献

1. Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*, ICLR 2025. [arXiv:2410.07864](https://arxiv.org/abs/2410.07864)
2. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023.
3. Open X-Embodiment Collaboration et al. *Open X-Embodiment*, 2023.


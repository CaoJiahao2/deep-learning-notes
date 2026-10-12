# RT-1：机器人 Transformer 的真机规模化起点

> RT-1 用 13 万条真实机器人轨迹证明：一个多任务 Transformer 策略可以在真实厨房环境中泛化，而不是只在仿真或单任务上工作。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | RT-1: Robotics Transformer for Real-World Control at Scale (2022) |
| 机构 | Google Robotics / Google Brain / Everyday Robots |
| 关键词 | language-conditioned policy、action discretization、real-world scaling |
| 定位 | 现代 VLA 与机器人基础模型的开山工作 |

---

## 1 背景与问题

在 RT-1 之前，机器人模仿学习常见两种模式：

1. 单任务策略：数据效率高，但无法组合任务；
2. 多任务策略：多为小规模或仿真实验，难以证明真实世界 scaling。

机器人数据采集昂贵、动作空间异构、任务标注质量参差，导致“Foundation Model 路线”是否能迁移到机器人上并不清楚。RT-1 的目标是用真实机器人、真实环境、真实语言指令做一次规模验证。


---

## 2 核心贡献

1. 构建超过 130k episodes、700+ 指令的真实机器人数据集；
2. 提出 RT-1 架构：FiLM-EfficientNet 视觉编码、TokenLearner 压缩、Transformer 解码离散动作；
3. 系统评估任务、指令、背景、干扰物、长程组合等泛化维度；
4. 验证混合仿真数据和跨机器人数据可以进一步提升泛化；
5. 确立“语言条件 + 离散动作 token + 大规模真机数据”的技术路线。


---

## 3 核心原理与方法详解

RT-1 学习条件策略：

$$
a_t \sim \pi_\theta(a_t\mid I_t, \ell),
$$

其中 $I_t$ 是当前图像，$\ell$ 是自然语言指令。连续动作先被离散化：

$$
a_t^{(d)} \in \{0,1,\dots,K-1\},
$$

每个维度独立 bin 化，再作为特殊 token 进入输出词表。

模型流程：

```text
图像 + 语言指令
    ↓
FiLM-conditioned EfficientNet-B3
    ↓
TokenLearner 压缩视觉 token
    ↓
Transformer decoder
    ↓
离散 action token
    ↓
反量化为底盘 / 机械臂 / 终止动作
```

动作包含机械臂位移、底盘位移和终止信号，运行频率约 3 Hz。


---

## 4 数据与训练

- 13 台真实机器人、约 17 个月持续采集；
- 约 130k 成功演示、700+ 语言指令；
- 场景为办公室与厨房，包含抽屉、桌面、 receptacle 等真实物体；
- 数据按成功 episode 过滤，使用 imitation learning 训练；
- 论文做了数据量、数据多样性、模型规模、架构组件和异构数据混合的消融；
- RT-1 参数量约 35M，因此能在机器人机载硬件上实时推理。


---

## 5 实验、结果与解读

在约 3000 次真实机器人评测中，RT-1 的关键结果为：

| 场景 | RT-1 | 结论 |
|---|---:|---|
| 已见任务 | 约 97% | 多任务学习没有牺牲训练任务性能 |
| 未见指令 | 约 76% | 语言组合泛化明显 |
| 干扰物 | 约 83% | 对视觉干扰较稳 |
| 新背景 | 约 59% | 场景级分布变化仍难 |

RT-1 还能接入 SayCan 风格的高层规划器完成长程任务。加入仿真数据和 Kuka 数据可以提升泛化，同时基本保持原任务表现。


---

## 6 精读要点与实践建议

- **核心结论**：机器人策略的 scaling 依赖“数据规模 + 任务多样性”，不是简单堆参数；
- **动作离散化**：让 Transformer 直接输出动作，工程简单，但为后来 RT-2 / OpenVLA 的 token 路线埋下伏笔；
- **TokenLearner**：视觉 token 压缩对机载推理至关重要；
- **数据组织比单条轨迹更重要**：任务覆盖与语言标注决定可组合性；
- **与经典 BC 的区别**：RT-1 是任务无关的通用骨干，而不是每个任务单独训练。


---

## 7 局限与后续影响

- 真机数据成本极高，130k episodes 相比互联网数据仍然极小；
- 离散动作损失精度，难以输出高频、平滑、多峰轨迹；
- 3 Hz 控制不适合高动态任务；
- 语义知识完全来自机器人数据和语言条件，没有互联网图文预训练；
- 后续 RT-2 引入 VLM co-training，π0 引入 Flow Matching 连续动作专家。


---

## 相关阅读

- [SayCan 论文精读](./saycan.md)
- [RT-2 论文精读](./rt-2.md)
- [RT-X 论文精读](./rt-x.md)
- [VLA 模型总览](../vla.md)

---

## 参考文献

1. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022. [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
2. Ahn et al. *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, CoRL 2022.
3. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023.


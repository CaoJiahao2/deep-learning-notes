# RT-X / Open X-Embodiment：跨机器人数据集与通用策略

> RT-X 把 21 家机构的 22 种机器人数据放进同一格式，证明跨 embodiment 联合训练可以产生正迁移。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models (2023) |
| 机构 | Open X-Embodiment Collaboration / Google DeepMind 等 |
| 关键词 | cross-embodiment、open dataset、positive transfer、robot foundation model |
| 定位 | 机器人基础模型开源数据生态的里程碑 |

---

## 1 背景与问题

机器人学习长期存在“数据孤岛”：不同实验室的机器人、相机、动作空间、任务定义和数据格式互不兼容。即使某实验室有大量数据，也很难与其他数据合并训练。

Open X-Embodiment 的目标是回答：把多个机器人、多个场景、多种任务的数据统一后，能否训练一个可迁移的 generalist policy？


---

## 2 核心贡献

1. 汇集 22 种机器人、21 个机构的数据，覆盖 527 类技能、160266 个任务；
2. 提出统一数据格式与标准化接口；
3. 训练 RT-X 模型，包括 RT-1-X 和 RT-2-X；
4. 系统验证跨 embodiment 联合训练的正迁移；
5. 开源数据生态推动 OpenVLA、Octo、π 系列等后续基础模型发展。


---

## 3 核心原理与方法详解

RT-X 的关键技术不是单一新结构，而是把异构轨迹映射到统一训练接口：

```text
原始机器人轨迹
    ↓ 标准化观测：图像 / 相机视角 / 语言指令
    ↓ 标准化动作：末端位姿增量 / 夹爪 / 底盘
    ↓ 元数据：embodiment、场景、任务标签
统一 Open X-Embodiment 样本
```

对于同一骨干模型，可以做两类训练：

- **单机器人数据**：$D_i$；
- **混合跨机器人数据**：$D=D_1\cup D_2\cup\cdots\cup D_M$。

若混合训练在机器人 $i$ 上的成功率高于只用 $D_i$，说明存在正迁移。


---

## 4 数据与训练

- 数据覆盖桌面操作、抽屉、厨房、仓储等场景；
- 处理不同相机数量、图像分辨率、控制频率和动作语义；
- 对指令进行规范化，同时保留原始任务描述；
- RT-1-X 基于 RT-1 架构，RT-2-X 基于 RT-2 / 大型 VLA 架构；
- 评测在多个原始机器人和新任务上进行，避免只看训练集指标；
- 社区后续形成 OXE 格式，成为许多开源 VLA 的数据加载基础。


---

## 5 实验、结果与解读

论文发现，跨机器人混合训练可以在多个平台上提升表现：

- RT-1-X 相比对应单机器人 RT-1 基线，在多平台平均成功率有显著提升；
- RT-2-X 在跨 embodiment 语义泛化上超过原始 RT-2；
- 不同机器人之间能共享“接近物体、抓取、放置、打开、移动”等操作知识；
- 但迁移并非自动发生，动作归一化、数据平衡和任务语义对齐非常关键；
- 数据规模扩大也带来长尾任务覆盖和评测难度的挑战。


---

## 6 精读要点与实践建议

- **数据格式优先于模型花活**：没有统一 schema，跨机器人训练无法复现；
- **embodiment gap 是工程问题也是学习问题**：相同任务语义可以有完全不同动作参数；
- **正迁移来自共享结构**：视觉、语言、任务语义、操作顺序比电机细节更容易共享；
- **开源数据集改变了研究范式**：OpenVLA 与后续模型都建立在 OXE 生态上；
- **实践建议**：先统一动作语义与指令词汇，再考虑混合训练。


---

## 7 局限与后续影响

- 不同机器人硬件差异仍会造成负迁移，需要平衡与过滤；
- 数据质量、任务标注和成功定义不一致；
- 覆盖以桌面操作为主，接触密集和双臂任务仍不足；
- 模型对高频控制和非结构化场景仍有限制；
- 后续 DROID、AgiBot World、LeRobot 等数据集继续扩充规模与真实性。


---

## 相关阅读

- [RT-1 论文精读](./rt-1.md)
- [RT-2 论文精读](./rt-2.md)
- [OpenVLA 论文精读](./openvla.md)
- [数据集与数据采集](../data.md)

---

## 参考文献

1. Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2023/2025. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)
2. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022.
3. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024.


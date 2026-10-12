# RT-Trajectory：用轨迹草图桥接任务泛化

> 语言擅长说明“做什么”，轨迹草图补充“怎么走”；RT-Trajectory 用粗粒度 2D 曲线作为中间任务表示，让策略泛化到语言难以覆盖的新运动。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches (2023) |
| 机构 | Google DeepMind / UC San Diego / Stanford / Intrinsic |
| 关键词 | trajectory sketch、hindsight labeling、mid-level conditioning |
| 定位 | RT 系列中“任务表示”路线的重要分支 |

---

## 1 背景与问题

语言指令有两个矛盾：太短则缺少运动细节，太长则人类难以描述。目标图像能说明结果，却难以表达过程，也容易在测试时构造。

RT-Trajectory 的观察是：一个粗略的末端执行器轨迹草图既足够具体，能指导运动；又足够抽象，允许策略根据视觉场景解释它。


---

## 2 核心贡献

1. 提出 trajectory sketch 作为策略条件；
2. 用 hindsight labeling 从已有演示中自动生成训练草图，不需要人工标注；
3. 设计 2D 与 2.5D 轨迹图像表示，编码时间、高度和夹爪交互；
4. 支持人类手绘、人类视频、生成模型、 waypoint 生成器等多种草图来源；
5. 在相同训练数据下扩展语言条件策略的任务范围。


---

## 3 核心原理与方法详解

训练时先从每条演示的末端 3D 位置 $p_t$ 投影到相机平面：

$$
u_t = \Pi K [R\mid t] p_t,
$$

再把 $u_t$ 连成曲线并编码额外信息：

- **颜色渐变**：红色通道表示归一化时间；
- **高度**：绿色通道编码末端相对高度，形成 2.5D 草图；
- **交互标记**：绿色/蓝色圆点标注夹爪闭合与释放；
- **图像条件**：策略输入为当前观测 $I_t$ 与草图 $c_{\text{traj}}$。

策略学习：

$$
a_t \sim \pi_\theta(a_t\mid I_t, c_{\text{traj}}).
$$

关键在于草图是从演示 hindsight 生成的，不需要人在演示时额外画出目标轨迹。


---

## 4 数据与训练

- 基于 RT-1 风格 Transformer 策略与 imitation learning；
- 训练数据复用已有机器人演示，通过后处理自动提取轨迹标签；
- 测试时草图可来自四种来源：

```text
人工手绘 → 简单直观
人类视频 → 迁移人的运动
生成模型 → 从语言/图像生成草图
waypoint 模型 → 先出关键点再连曲线
```

- 对静态相机与固定机械臂设定做投影，假设机器人基座与相机外参已知；
- 训练目标与原始离散动作 imitation learning 一致。


---

## 5 实验、结果与解读

真实机器人实验显示：

- 轨迹草图条件策略能执行语言条件和目标图像条件难以覆盖的新任务；
- 2.5D 草图比纯 2D 更适合需要高度变化的任务；
- 交互标记对抓取/释放时机有帮助；
- 自动生成的草图与人类绘制草图均可使用；
- 泛化来自任务表示更接近训练分布，而不是模型学到全新物理能力。


---

## 6 精读要点与实践建议

- **中间表示是关键**：语言、草图、目标图像、latent plan 各有表达力边界；
- **hindsight relabeling** 是低成本扩充监督的通用技巧；
- **2.5D 表示**用简单通道编码几何信息，适合缺乏 3D 输入的系统；
- **实践建议**：当任务语义已知但运动路径重要时，加入轨迹条件比堆数据更有效；
- **与 RT-H 的关系**：RT-Trajectory 用图像草图，RT-H 用语言化的中间动作描述。


---

## 7 局限与后续影响

- 草图获取依赖相机标定和末端状态；
- 2D 投影可能丢失遮挡与深度歧义；
- 对动态场景和移动操作不够通用；
- 草图质量直接影响策略上限；
- 后续 RT-H、HiRT、π0.5 更强调语言子任务或高层策略输出。


---

## 相关阅读

- [RT-1 论文精读](./rt-1.md)
- [RT-H 论文精读](./rt-h.md)
- [操作学习](../manipulation.md)

---

## 参考文献

1. Gu et al. *RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches*, ICLR 2024. [arXiv:2311.01977](https://arxiv.org/abs/2311.01977)
2. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022.
3. Belkhale et al. *RT-H: Action Hierarchies Using Language*, 2024.


# RT-H：用语言动作层级分解机器人控制

> RT-H 在高层任务和低层动作之间插入一层“语言化中间动作”，让模型先预测做什么动作，再预测如何执行，从而提升数据效率和长程泛化。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | RT-H: Action Hierarchies Using Language (2024) |
| 机构 | Google DeepMind / UC Berkeley |
| 关键词 | action hierarchy、language actions、hindsight relabeling、multi-task policy |
| 定位 | RT 系列的分层表示代表作 |

---

## 1 背景与问题

RT-1 / RT-2 直接从任务语言到动作，容易把不同任务中的相同运动模式重复学习。例如“拿苹果”和“拿杯子”的低层动作相似，但语言完全不同。

RT-H 的思路是拆成两级：

1. **language action**：描述当前动作，如“向左移动机械臂并张开夹爪”；
2. **robot action**：具体位移、夹爪与终止信号。

中间层让语义相似的数据共享运动知识。


---

## 2 核心贡献

1. 提出语言动作层级（action hierarchy using language）；
2. 从已有轨迹自动 hindsight 标注 language action，无需人工重新演示；
3. 训练一个模型同时预测中间语言动作和低层动作；
4. 在推理时可人工干预中间动作，实现人机可控纠错；
5. 在多任务、长程任务和少量数据设定下优于 RT-1 / RT-2 风平整端映射。


---

## 3 核心原理与方法详解

RT-H 将策略分解为：

$$
L_t \sim p_\theta(L_t\mid o_{\le t}, \ell, L_{<t}),
$$

$$
a_t \sim p_\theta(a_t\mid o_{\le t}, \ell, L_{<t}, L_t).
$$

其中 $L_t$ 是 language action，$a_t$ 是低层动作。训练目标为两级负对数似然相加：

$$
\mathcal{L}
=
-\log p_\theta(L_t\mid o_{\le t},\ell,L_{<t})
-\log p_\theta(a_t\mid o_{\le t},\ell,L_{<t},L_t).
$$

两级共享视觉和语言表征，但中间动作强制模型显式解释“为什么这样做”。


---

## 4 数据与训练

- 使用 VLM 或规则自动从轨迹状态生成 language action；
- 保留原始机器人演示，无需额外遥控标注；
- 对“移动方向、夹爪开合、接近/离开、终止”等中间语义建立词表；
- 低层动作仍可用离散 token 输出；
- 训练时可随机屏蔽或保留真实 language action，让模型同时学习预测和条件执行；
- 支持人在推理时修改中间动作，从而干预策略。


---

## 5 实验、结果与解读

论文在真实机器人任务上比较 RT-H 与直接任务到动作的基线，结果显示：

- 中间动作使新任务和长程任务成功率提升；
- 少量演示数据下，数据效率更高；
- 自动标注的 language action 质量足以支撑训练；
- 人工修改中间动作可以纠正部分执行错误；
- 层级化不会明显牺牲控制频率，因为两层都在同一模型上下文中生成。


---

## 6 精读要点与实践建议

- **中间表示价值**：比任务语言更接近动作，比原始动作更可迁移；
- **hindsight 语言标注**能把已有数据转化为层级监督；
- **可干预性**是分层系统的重要工程优势；
- **与现代 VLA 的关系**：π0.5 的高层子任务、Helix 的 S2/S1、Gemini Robotics 的推理与控制耦合都可视为不同形式的层级化；
- **实践建议**：低层数据稀缺但任务语言丰富时，先做中间动作标注往往比直接微调更有效。


---

## 7 局限与后续影响

- language action 词表需要设计，覆盖不足会成为瓶颈；
- 自动标注可能产生歧义或错误描述；
- 中间层与低层动作的错误会级联；
- 主要面向桌面操作，复杂接触任务仍困难；
- 连续多峰动作分布仍需 Diffusion / Flow Matching 表达。


---

## 相关阅读

- [RT-2 论文精读](./rt-2.md)
- [RT-Trajectory 论文精读](./rt-trajectory.md)
- [π0 论文精读](./pi-0.md)

---

## 参考文献

1. Belkhale et al. *RT-H: Action Hierarchies Using Language*, 2024. [arXiv:2403.01823](https://arxiv.org/abs/2403.01823)
2. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023.
3. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.


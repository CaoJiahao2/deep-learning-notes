# RT-2：Vision-Language-Action 模型的正式确立

> RT-2 把机器人动作写成文本 token，与互联网图文数据共同微调，使 VLM 的网页知识第一次真正迁移到真实机器人控制。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023) |
| 机构 | Google DeepMind |
| 关键词 | VLA、co-fine-tuning、action token、emergent reasoning |
| 定位 | VLA 术语与范式的奠基论文 |

---

## 1 背景与问题

RT-1 证明了真机多任务规模化有效，但它缺乏互联网级语义先验。对于“把最累的人想喝的饮料拿给我”这类指令，机器人数据中几乎不可能覆盖所有组合。

RT-2 的假设是：如果视觉语言模型已经从网页学到物体语义、常识与关系推理，只要把动作表示成同一词表中的 token，这些知识就能通过端到端训练迁移到控制。


---

## 2 核心贡献

1. 正式提出 **Vision-Language-Action model** 类别；
2. 将动作离散化为特殊文本 token，使 VLM 能同时输出答案和机器人动作；
3. 在 VQA / 图文数据和机器人轨迹上 co-fine-tune，保持通用多模态能力；
4. 展示语义涌现能力：新物体、新指令、数量/类别/空间关系、简易常识推理；
5. 引入 chain-of-thought 形式的多阶段语义推理；
6. 证明大模型可以既做高层语义理解，又直接输出低层控制。


---

## 3 核心原理与方法详解

RT-2 将控制写成条件生成：

$$
y \sim p_\theta(y\mid I_t, q),
$$

当问题是 VQA 时，$y$ 是自然语言；当问题是“机器人下一步做什么”时，$y$ 是 action token 序列：

```text
连续动作 → 每维 256 bins → 特殊 action token
        → 与语言共享自回归输出空间
        → de-tokenize 得到底盘 / 机械臂动作
```

模型基于大型 VLM（论文报告 PaLI-X 与 PaLM-E 路线的大规模实例），训练目标与语言模型一致：

$$
\mathcal{L}
= -\sum_i \log p_\theta(y_i\mid y_{<i}, I_t, q).
$$

因此 VQA 和动作预测在数学上都是下一个 token 预测。


---

## 4 数据与训练

- **机器人数据**：基于 RT-1 风格的真实演示轨迹；
- **网页数据**：视觉问答、图文描述等互联网级任务；
- **co-fine-tuning**：不是先预训练再完全冻结，而是在两类任务上共同微调；
- **动作词表**：与语言词表合并，模型可以把“语义答案”和“动作命令”视为同一输出形式；
- **评测规模**：论文报告约 6000 次真实机器人试验；
- **推理频率**：大型 VLM 导致低频控制，论文通过动作离散与系统设计弥补一部分延迟。


---

## 5 实验、结果与解读

RT-2 的核心实验结论：

- 在未见物体和语义指令上显著优于 RT-1；
- 能执行机器人数据中未出现的指令，如数字、图标、抽象类别；
- 能处理“拿最小的物体”“拿离某物最近的物体”等关系推理；
- chain-of-thought 可支持多阶段推理，例如选择可临时当锤子的物体；
- 语义泛化提升明显，但精细节奏控制与高频动作仍不如后来的连续动作头路线。


---

## 6 精读要点与实践建议

- **读法重点**：RT-2 的贡献不是发明 action token，而是证明网页知识与动作输出可以在同一模型中共训练；
- **工程启示**：当机器人数据无法覆盖语义长尾时，VLM co-training 是有效放大器；
- **表示折中**：离散 token 借用 LLM 栈，但牺牲动作精度与多峰建模能力；
- **系统启示**：VLA 的“大脑”能力来自 VLM，控制质量还取决于动作头；
- **与 PaLM-E 的关系**：PaLM-E 是具身推理，RT-2 把输出端推进到动作。


---

## 7 局限与后续影响

- 模型规模巨大、闭源，训练成本高；
- 自回归动作输出延迟高，不适合精细动态任务；
- 离散 bin 限制高频连续控制；
- 网页数据与机器人观测之间存在 domain gap；
- 后续 OpenVLA 证明开源 7B 模型可用，π0 / RDT-1B 则转向 Diffusion / Flow Matching 连续动作。


---

## 相关阅读

- [RT-1 论文精读](./rt-1.md)
- [PaLM-E 论文精读](./palm-e.md)
- [OpenVLA 论文精读](./openvla.md)
- [VLA 模型总览](../vla.md)

---

## 参考文献

1. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
2. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022.
3. Driess et al. *PaLM-E: An Embodied Multimodal Language Model*, ICML 2023.


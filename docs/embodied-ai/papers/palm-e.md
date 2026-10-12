# PaLM-E：具身多模态语言模型

> PaLM-E 把图像、状态和传感器数据当作连续 embedding 注入 LLM，让同一个模型同时做场景推理、任务规划和机器人控制。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | PaLM-E: An Embodied Multimodal Language Model (2023) |
| 机构 | Google / TU Berlin |
| 关键词 | multimodal LLM、embodied reasoning、continuous observation embedding |
| 定位 | VLM 知识进入机器人系统的关键桥梁 |

---

## 1 背景与问题

PaLM-E 面对的问题是：纯文本 LLM 无法理解机器人观测，而早期视觉语言模型又缺少长上下文与复杂推理能力。机器人任务需要同时具备：

- 视觉场景理解；
- 空间关系推理；
- 任务规划；
- 与低层控制接口对接。

PaLM-E 的路线不是把视觉模型作为外挂工具，而是把多模态观测作为 token 序列直接嵌入 LLM。


---

## 2 核心贡献

1. 提出统一的多模态具身 LLM：文本、图像、机器人状态、场景表示可以交织输入；
2. 展示连续多模态 embedding 能与语言 token 共同训练，而不是只在提示层拼接；
3. 通过 co-training 保持通用语言能力，同时获得机器人场景推理能力；
4. 为 RT-2 的“VLM 直接输出动作 token”奠定基础。


---

## 3 核心原理与方法详解

PaLM-E 基于 PaLM，将观测 $o_t$ 编码为向量序列：

$$
H_{\text{LM}} = f_\theta(x_{<t}, E_v(I_t), E_s(s_t), x_{>t}),
$$

其中 $E_v(I_t)$ 是图像编码器，$E_s(s_t)$ 是机器人状态投影。图像 token 与文本 token 共同进入 Transformer 上下文。

关键设计：

- **多模态 token 化**：图像可被切成多个 patch token，状态通过 MLP 投影；
- **交错序列**：模型可以输入“图像、问题、图像、状态、问题”的混合序列；
- **co-training**：机器人任务与互联网文本/图文数据混合训练，防止语言能力崩塌；
- **任务输出**：可以直接输出自然语言规划，或与低层技能策略连接。


---

## 4 数据与训练

- 使用预训练 ViT 或同类视觉编码器，训练时保持部分模块冻结以降低成本；
- 机器人状态包括位姿、关节状态、目标位置等连续值；
- 在机器人数据与通用多模态数据之间做混合采样；
- 训练目标仍是自回归语言建模，只是 token 序列扩展到多模态输入；
- 实验覆盖移动操作、桌面任务与多模态推理数据集。


---

## 5 实验、结果与解读

PaLM-E 展示了较强的多模态推理能力，例如根据图像回答物体关系、判断任务下一步、生成子任务序列。机器人实验表明：

- 与仅用文本 LLM 相比，视觉输入显著减少对环境状态的误判；
- 多模态 co-training 能保留通用推理能力；
- 更大的语言模型规模带来更好的推理与任务规划表现；
- 但模型本身不是高频控制器，最终动作仍需技能策略执行。


---

## 6 精读要点与实践建议

- **读法重点**：PaLM-E 的“E”是 embodied input，而不是 end-to-end action；
- **工程启示**：多模态 LLM 适合当“大脑”，低层策略仍需专门设计；
- **表示设计**：连续状态投影成 token 是处理异构传感器的通用方法；
- **与 SayCan 的区别**：SayCan 的 LLM 只见文本，PaLM-E 让 LLM 直接看见场景；
- **与 RT-2 的区别**：PaLM-E 输出规划，RT-2 把动作也纳入同一输出词表。


---

## 7 局限与后续影响

- 模型大、推理延迟高，不适合直接闭环控制；
- 图像 token 数量带来长上下文压力；
- 对空间几何的理解仍弱于显式 3D 表示；
- 数据与算力门槛高；
- 后续 RT-2 用更明确的 VLA 训练目标验证了动作知识迁移，π 系列则转向“VLM 骨干 + 连续动作专家”。


---

## 相关阅读

- [SayCan 论文精读](./saycan.md)
- [RT-2 论文精读](./rt-2.md)
- [多模态对齐与融合](../../llm-mllm/multimodal-alignment.md)

---

## 参考文献

1. Driess et al. *PaLM-E: An Embodied Multimodal Language Model*, ICML 2023.
2. Ahn et al. *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, CoRL 2022.
3. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023.


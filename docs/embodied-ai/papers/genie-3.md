# Genie 3：实时可交互的通用世界模型

> Genie 3 能从文本生成 720p、24 FPS、可实时导航并保持数分钟一致性的交互世界，把基础世界模型从“生成视频”推进到“可探索环境”。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 技术报告 | Genie 3: A new frontier for world models (2025) |
| 机构 | Google DeepMind |
| 关键词 | real-time world model、720p 24 FPS、promptable world events、embodied agent research |
| 定位 | DeepMind Genie 系列截至 2025 年的公开前沿版本 |

---

## 1 背景与问题

Genie 证明了无标注视频可以学出交互环境，但存在时长、分辨率和实时性问题。Genie 2 扩展到 3D 世界，Genie 3 则进一步要求：

- 用户动作实时改变世界；
- 环境状态跨分钟保持一致；
- 文本可以触发世界事件；
- 生成的世界足以用于 agent 训练与评测。

这使世界模型从生成媒体更接近无限课程环境。


---

## 2 核心贡献

1. 文本 prompt 生成可实时交互的动态世界；
2. 达到 720p、24 FPS、数分钟一致性；
3. 支持用户导航输入与 promptable world events；
4. 显式处理自回归生成的长时视觉记忆，可回看约一分钟前的场景状态；
5. 与 SIMA agent 结合，将生成世界作为 agent 训练 / 评测环境；
6. 公开讨论 action space、其他 agent 交互、真实地理精度和时长等限制；
7. 以 limited research preview 方式提供访问。


---

## 3 核心原理与方法详解

Genie 3 的形式可抽象为动作条件自回归环境：

$$
x_{t+1}
\sim
p_\theta(x_{t+1}\mid x_{\le t}, a_t, u_t),
$$

其中 $a_t$ 是用户导航动作，$u_t$ 是文本世界事件 prompt。

关键难点：

- **实时性**：每秒生成多帧，同时处理新动作；
- **长时一致性**：重访一分钟前位置时，场景应保持一致；
- **可交互性**：用户输入改变相机轨迹、环境状态和事件；
- **世界先验**：从文本与海量视频学习中获得物体、物理和场景布局。

与 NeRF / 3D Gaussian Splatting 不同，Genie 3 不依赖显式重建的 3D 表示，而是逐帧根据描述与动作生成世界，因此更容易产生动态和虚构场景。


---

## 4 数据与训练

- 输入：文本 prompt 或初始图像 / 世界设定；
- 交互：导航动作、实时用户输入、文本触发世界事件；
- 输出：720p、24 FPS 的交互视频环境；
- 训练细节未完整公开，更接近系统技术报告而非可复现论文；
- 实验将 SIMA 的导航动作发送给 Genie 3，模型不知道 agent 目标，只根据动作模拟未来；
- 责任治理方面采用 limited research preview，面向学术与创作者小范围开放。


---

## 5 实验、结果与解读

官方报告的能力包括：

- 自然现象与环境交互：水、光照、风、植被等；
- 一致的导航体验：物体在进出视野后保持位置；
- promptable world events：用文本触发场景事件；
- 逼真、动画、虚构和历史场景生成；
- SIMA agent 可在 Genie 3 世界中执行更长动作序列和更复杂目标；
- 局限包括直接动作空间有限、多智能体交互困难、真实地理位置不准确、连续交互只有数分钟。


---

## 6 精读要点与实践建议

- **实时交互是世界模型的重要分水岭**，比离线视频生成更接近训练环境；
- **一致性是可用性核心**：不稳定世界无法用于策略学习；
- **文本事件接口**让非运动控制任务可以被表达；
- **与机器人关系**：短期可作为表征、规划与评测环境；直接迁移到真机还缺本体动作与接触物理；
- **实践建议**：用 Genie 3 类模型生成 curriculum 和 stress test，而不是直接替代高保真物理仿真；
- **评测建议**：一致性、可控性、物理错误率比单次视觉质量更重要。


---

## 7 局限与后续影响

- 没有公开完整模型与训练方法；
- 交互动作空间有限，不能执行复杂机器人操作；
- 多个独立 agent 的交互仍难准确建模；
- 只能维持数分钟，不支持小时级世界状态；
- 真实地点地理精度不足；
- 物理一致性与安全评估仍需外部验证。


---

## 相关阅读

- [Genie 论文精读](./genie.md)
- [Sora 2 世界模型精读](./sora-2.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. Parker-Holder & Fruchter, Google DeepMind. *Genie 3: A new frontier for world models*, 2025. [官方博客](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
2. Bruce et al. *Genie: Generative Interactive Environments*, 2024.
3. Google DeepMind. *SIMA: Generalist AI Agents for 3D Environments*, 2024/2025.


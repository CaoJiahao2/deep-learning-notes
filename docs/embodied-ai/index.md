# 具身智能

> 让智能走出屏幕，进入物理世界。

具身智能（Embodied AI）= 感知 + 决策 + 实体动作的闭环。本章节按「基础与策略 → 数据与仿真 → 世界模型与评测 → 前沿与应用」组织：基础与策略从问题建模（MDP/POMDP）讲到 VLA 与策略学习、机器人控制；数据与仿真解决「数据从哪来、如何便宜地来」；世界模型与评测回答「如何想象未来、如何衡量好坏」；前沿与应用覆盖灵巧操作、人形机器人与产业落地。

## 章节地图

<div class="grid cards" markdown>

-   :material-robot-industrial:{ .lg .middle } **基础与策略**

    ---

    问题建模与「大脑→小脑→肢体」的实现路径。

    - [具身智能基础](./fundamentals.md) — MDP/POMDP、三种范式、关键概念
    - [VLA 模型](./vla.md) — RT-1→RT-2→OpenVLA→π0→GR00T 演进
    - [模仿学习与策略学习](./policy-learning.md) — BC、Diffusion Policy、离线 RL
    - [机器人控制与运动规划](./control.md) — 运动学/动力学、MPC、全身控制

-   :material-database-outline:{ .lg .middle } **数据与仿真**

    ---

    具身智能的瓶颈在数据。

    - [数据集与数据采集](./data.md) — 数据金字塔、OXE/DROID/AgiBot World、数据飞轮
    - [Sim-to-Real 与仿真器](./sim2real.md) — Isaac Lab、Genesis、域随机化

-   :material-earth:{ .lg .middle } **世界模型与评测**

    ---

    想象未来与衡量进展。

    - [世界模型](./world-model.md) — Dreamer、Genie、Cosmos、与 VLA 融合
    - [具身智能评测基准](./evaluation.md) — CALVIN、RLBench、ManiSkill、SimplerEnv

-   :material-rocket-launch:{ .lg .middle } **前沿与应用**

    ---

    走向通用与落地。

    - [操作学习：灵巧操控与移动操作](./manipulation.md) — 接触丰富操作、灵巧手、WAM
    - [具身智能前沿专题](./frontier.md) — 分层规划、双系统 VLA、Humanoid
    - [具身智能应用与产业版图](./applications.md) — 工业/物流/家庭场景、产业格局

</div>

## 学习建议

1. **先建模再谈模型**：[具身智能基础](./fundamentals.md) 中的 POMDP 框架与三种范式对比是全章的坐标系，后续所有模型（VLA、Diffusion Policy、世界模型）都能放回这个坐标系。
2. **沿 VLA 主线读**：[VLA 模型](./vla.md) → [策略学习](./policy-learning.md) → [操作学习](./manipulation.md) 构成「架构 → 训练方法 → 任务细化」的递进线。
3. **数据决定上限**：具身智能当前的瓶颈更多在数据侧，[数据集与数据采集](./data.md) 与 [Sim-to-Real](./sim2real.md) 值得精读。
4. **模型侧背景**：VLA 本质是 MLLM + 动作头，建议先读 [多模态对齐与融合](../llm-mllm/multimodal-alignment.md) 与 [Flow Matching](../architectures/flow-matching.md)（π0 的动作生成范式）。

## 延伸阅读

- [统一多模态与任意到任意生成](../llm-mllm/unified-multimodal.md) — VLA 的多模态底座
- [强化学习与对齐](../training/reinforcement-learning.md) — 策略学习的 RL 基础
- [常用数据集](../resources/datasets.md) — 具身智能数据集速查

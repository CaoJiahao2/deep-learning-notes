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

-   :material-book-open-page-variant:{ .lg .middle } **经典论文与新模型精读**

    ---

    每篇论文单独一页，补齐总览页无法展开的细节。

    - [论文地图](./papers/index.md) — RT / π / 最新 VLA / 世界模型阅读路线
    - [RT 系列](./papers/rt-1.md) — RT-1、RT-2、RT-X、RT-Trajectory、RT-H
    - [π 系列](./papers/pi-0.md) — π0、π0.5、π*0.6、π0.7
    - [最新 VLA](./papers/gemini-robotics.md) — OpenVLA、RDT、GR00T、Gemini、Helix、Xiaomi
    - [世界模型](./papers/cosmos.md) — Dreamer、Genie、Cosmos、V-JEPA、GE 系列

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
5. **精读原始论文**：总览页负责建立坐标，[论文地图](./papers/index.md) 负责深入单篇。VLA 建议按 RT-1 → RT-2 → π0 → π0.5 → π*0.6 → π0.7 阅读；世界模型建议按 World Models → DreamerV3 → Genie → Cosmos → GE 系列阅读。

## 延伸阅读

- [统一多模态与任意到任意生成](../llm-mllm/unified-multimodal.md) — VLA 的多模态底座
- [具身智能论文地图](./papers/index.md) — 26 篇经典与前沿论文单页精读
- [强化学习与对齐](../training/reinforcement-learning.md) — 策略学习的 RL 基础
- [常用数据集](../resources/datasets.md) — 具身智能数据集速查

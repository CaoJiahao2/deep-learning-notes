# 训练技术

> 架构决定上限，训练决定你能到达的位置。

本章节按「训练基础 → 学习范式 → 规模化训练」组织：训练基础覆盖从数据准备到收敛的完整链路（流程、反向传播、优化器、正则化）；学习范式介绍如何在没有标注或只有奖励时学习（自监督、强化学习）；规模化训练解决模型大到单卡放不下之后的问题（分布式策略、集群基础设施）。

## 章节地图

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **训练基础**

    ---

    一次完整训练的解剖学。

    - [完整训练流程](./training-pipeline.md) — 从数据准备到模型部署全链路
    - [反向传播](./backpropagation.md) — 链式法则、autograd、梯度检查
    - [优化器详解](./optimization.md) — SGD→AdamW→Lion 演进与调参
    - [正则化技术](./regularization.md) — Dropout、BN/LN、数据增强

-   :material-refresh:{ .lg .middle } **学习范式**

    ---

    标注稀缺时代的两把钥匙。

    - [自监督学习](./self-supervised-learning.md) — SimCLR/MoCo、MAE、DINOv2、CLIP
    - [强化学习与对齐](./reinforcement-learning.md) — MDP→PPO→RLHF/DPO→GRPO

-   :material-server:{ .lg .middle } **规模化训练**

    ---

    当模型大到单卡放不下。

    - [分布式训练](./distributed.md) — DP、TP、PP、ZeRO、DeepSpeed
    - [AI 训练基础设施](./ai-infrastructure.md) — GPU 集群、RDMA、容错、成本模型

</div>

## 学习建议

1. **按流程顺序读**：训练基础四篇按真实项目的推进顺序组织（先流程、再求导、再调优、再防过拟合），建议顺序通读。
2. **反向传播要动手推**：[反向传播](./backpropagation.md) 中的推导建议在纸上完成一遍，并用梯度检查代码验证，这是理解一切训练 trick 的基础。
3. **RL 先建立 MDP 直觉**：[强化学习与对齐](./reinforcement-learning.md) 是后训练（RLHF/DPO/GRPO）的理论入口，读 [后训练综述](../llm-mllm/post-training-survey.md) 前务必过一遍。
4. **分布式结合实验体会**：单机多卡跑通一次 DeepSpeed ZeRO 配置，再读 [分布式训练](./distributed.md) 会有完全不同的体感。

## 延伸阅读

- [优化理论与方法](../fundamentals/optimization-theory.md) — 优化器的数学背景
- [预训练](../llm-mllm/pretraining.md) — 训练技术在 LLM 语境下的完整应用
- [AI 训练基础设施](./ai-infrastructure.md) — 训练侧的工程极点，与推理侧的 [推理服务化](../deployment/model-serving.md) 对照

# 大模型专题

> 从预训练到后训练，从语言到多模态，从能力扩展到安全治理 —— LLM 的完整知识版图。

本章节是全站最大的专题，按「基础与预训练 → 后训练与对齐 → 推理与能力扩展 → 多模态 → 评测与治理 → 前沿专题」六条线组织。前两条线回答「大模型是怎么造出来的」；推理与能力扩展回答「怎么用得更好」；多模态把视觉、听觉接入语言模型；评测与治理关注怎么知道模型好不好、可不可控；前沿专题覆盖代码智能与新型生成范式。

## 章节地图

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle } **基础与预训练**

    ---

    大模型的原材料与炼成方式。

    - [LLM/MLLM 全景调研](./survey.md) — 开源与闭源模型一览
    - [预训练](./pretraining.md) — 预训练目标、数据、Scaling Laws
    - [分词与词嵌入](./tokenizer-embedding.md) — BPE、WordPiece、SentencePiece
    - [大模型数据工程](./data-engineering.md) — 语料清洗、质量过滤、合成数据

-   :material-tune:{ .lg .middle } **后训练与对齐**

    ---

    从基座到可用的助手。

    - [微调技术](./fine-tuning.md) — LoRA、QLoRA、Adapter
    - [后训练 Post-Training 综述](./post-training-survey.md) — SFT、RLHF/DPO、GRPO、蒸馏
    - [LLM 安全与对齐](./safety-alignment.md) — 红队、Constitutional AI、可扩展监督

-   :material-rocket-launch:{ .lg .middle } **推理与能力扩展**

    ---

    同一个模型，换一种用法就是另一种能力。

    - [推理优化](./inference.md) — KV Cache、Flash Attention、量化
    - [提示工程](./prompt-engineering.md) — Zero/Few-shot、CoT、ReAct
    - [推理模型 o1/R1](./reasoning-models.md) — 长思维链、Test-time Compute、DeepSeek-R1
    - [测试时计算与推理扩展](./test-time-compute.md) — Best-of-N、ToT、搜索增强
    - [长上下文与检索增强](./long-context-rag.md) — 位置编码外推、RAG、GraphRAG

-   :material-image-outline:{ .lg .middle } **多模态**

    ---

    语言模型睁开眼睛、竖起耳朵。

    - [多模态对齐与融合](./multimodal-alignment.md) — CLIP、Q-Former、LLaVA
    - [多模态推理模型 LMRM 综述](./multimodal-reasoning-survey.md) — Perception→CoT→RL→Agent
    - [统一多模态与任意到任意生成](./unified-multimodal.md) — 统一分词器、Any-to-Any

-   :material-scale-balance:{ .lg .middle } **评测与治理**

    ---

    知道模型好在哪、坏在哪、可控吗。

    - [评测体系](./evaluation.md) — MMLU/GSM8K、LLM-as-Judge、评测陷阱
    - [大模型典型缺陷](./llm-defects.md) — 幻觉、灾难性遗忘、模型坍塌
    - [机制可解释性](./interpretability.md) — SAE、Circuits、Steering Vectors
    - [模型编辑与知识更新](./model-editing.md) — ROME/MEMIT、持续预训练
    - [个性化与定制化](./personalization.md) — 用户画像、个性化 RLHF

-   :material-flask:{ .lg .middle } **前沿专题**

    ---

    正在发生的研究方向。

    - [代码大模型与程序推理](./code-intelligence.md) — FIM、SWE-bench、Agentic Coding
    - [非自回归与扩散式语言模型](./non-autoregressive.md) — 投机解码、LLaDA、并行生成

</div>

## 学习建议

1. **先建全景再深入**：先读 [全景调研](./survey.md) 了解模型版图，再沿「预训练 → 微调 → 后训练综述」的主线走一遍造模型流程。
2. **推理侧与训练侧并重**：[推理优化](./inference.md) 与 [推理实现详解](../deployment/inference-implementation.md) 互为表里，前者讲算法、后者讲系统。
3. **评测先行**：做任何实验前先读 [评测体系](./evaluation.md)，避免被基准污染或指标误导。
4. **跟踪前沿**：[推理模型](./reasoning-models.md) 与 [测试时计算](./test-time-compute.md) 是当前变化最快的方向，建议配合 [学习资源](../resources/learning-resources.md) 里的论文渠道持续跟进。

## 延伸阅读

- [深度学习全景手册](../fundamentals/panorama.md) — LLM 在整个深度学习版图中的位置
- [智能体总览](../agents/index.md) — LLM 作为 Agent 大脑后的工程化
- [推理部署总览](../deployment/index.md) — 模型训练完成后的落地链路

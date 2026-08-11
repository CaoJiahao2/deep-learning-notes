# 深度学习笔记 — 文档索引

欢迎来到 deep-learning-notes 文档站。以下是按主题域组织的完整文档索引。

---

## 全景手册

- [深度学习基础知识全景手册](./fundamentals/panorama.md) — 从数学基础到大模型部署、计算机视觉、多模态与 VLA/世界模型的贯通式综述，附推荐学习顺序

## 入门指南

- [学习路线图](./getting-started/learning-roadmap.md) — 从零基础到独立研究的系统化学习路径
- [环境配置](./getting-started/environment-setup.md) — GPU 服务器、CUDA、PyTorch 环境搭建

## 基础理论

- [数学基础](./fundamentals/math-foundations.md) — 线性代数、微积分、概率与信息论（推导的地基）
- [机器学习概述](./fundamentals/machine-learning.md) — 基本概念、分类、三要素、评估方法

## 模型架构

- [Transformer 架构详解](./architectures/transformer.md) — Self-Attention、Multi-Head、位置编码
- [卷积神经网络 (CNN)](./architectures/cnn.md) — LeNet→ResNet→EfficientNet 演进
- [循环神经网络 (RNN/LSTM/GRU)](./architectures/rnn-lstm.md) — 序列建模与注意力机制
- [生成对抗网络 (GAN)](./architectures/gan.md) — 从 Vanilla GAN 到 StyleGAN
- [扩散模型 (Diffusion)](./architectures/diffusion.md) — DDPM、Stable Diffusion
- [Flow Matching 与 Rectified Flow](./architectures/flow-matching.md) — SD3、Flux、π0 的统一训练范式
- [视频生成](./architectures/video-generation.md) — Sora、Wan 2.1、DiT、3D VAE
- [可控图像生成](./architectures/controllable-image-generation.md) — ControlNet、IP-Adapter、InstantID、Leffa
- [目标检测](./architectures/object-detection.md) — 两阶段与单阶段（Faster R-CNN、YOLO）
- [图像分割](./architectures/segmentation.md) — 语义/实例分割（FCN、U-Net、Mask R-CNN、SAM）
- [单目深度与 3D 视觉](./architectures/monocular-depth.md) — Depth Anything v2、Metric3D v2
- [现代架构演进 MoE/SSM](./architectures/modern-architectures.md) — 稀疏专家、状态空间模型、高效注意力

## 训练技术

- [完整训练流程](./training/training-pipeline.md) — 从数据准备到模型部署
- [反向传播](./training/backpropagation.md) — 从链式法则到 softmax+交叉熵的完整推导
- [优化器详解](./training/optimization.md) — SGD→AdamW→Lion 演进
- [正则化技术](./training/regularization.md) — Dropout、BN/LN、数据增强
- [分布式训练](./training/distributed.md) — DP、TP、PP、ZeRO、DeepSpeed
- [强化学习与对齐](./training/reinforcement-learning.md) — MDP→PPO→RLHF/DPO→GRPO/RLVR

## 大模型专题

- [LLM/MLLM 全景调研](./llm-mllm/survey.md) — 开源与闭源模型一览
- [预训练](./llm-mllm/pretraining.md) — 预训练目标、数据工程、Scaling Laws
- [分词与词嵌入](./llm-mllm/tokenizer-embedding.md) — BPE、WordPiece、SentencePiece
- [微调技术](./llm-mllm/fine-tuning.md) — LoRA、QLoRA、Adapter、RLHF/DPO
- [推理优化](./llm-mllm/inference.md) — KV Cache、Flash Attention、量化
- [提示工程](./llm-mllm/prompt-engineering.md) — Zero/Few-shot、CoT、ReAct、提示注入
- [推理模型 o1/R1](./llm-mllm/reasoning-models.md) — CoT、Test-time Compute、DeepSeek-R1
- [多模态对齐与融合](./llm-mllm/multimodal-alignment.md) — CLIP、Q-Former、LLaVA、跨模态生成
- [评测体系](./llm-mllm/evaluation.md) — MMLU/GSM8K、LLM-as-Judge、评测陷阱
- [大模型缺陷](./llm-mllm/llm-defects.md) — 幻觉、灾难性遗忘、模型坍塌

## 智能体

- [AI Agent 基础与核心循环](./agents/fundamentals.md) — ReAct、规划范式、Agent 评测
- [工具使用与 MCP](./agents/tools-and-mcp.md) — Function Calling、Model Context Protocol
- [Agentic RAG](./agents/agentic-rag.md) — Agent 主导的检索与查询改写
- [Agent 记忆机制](./agents/memory.md) — 短期/长期、向量/笔记式、MemGPT
- [多智能体协作](./agents/multi-agent.md) — AutoGen、CrewAI、LangGraph
- [GUI Agent 与 Computer Use](./agents/gui-agent.md) — Operator、Computer Use、AutoGLM

## 具身智能

- [具身智能基础](./embodied-ai/fundamentals.md) — MDP/POMDP、三种范式、关键概念
- [VLA 模型](./embodied-ai/vla.md) — RT-2、OpenVLA、π0/π0.5、GR00T
- [模仿学习与策略学习](./embodied-ai/policy-learning.md) — BC、Diffusion Policy、Flow Matching Policy、DP3、HIL-SERL
- [世界模型](./embodied-ai/world-model.md) — Dreamer、Genie、GameNGen、UniSim
- [数据集与数据采集](./embodied-ai/data.md) — Open X-Embodiment、DROID、UMI、MimicGen
- [Sim-to-Real 与仿真器](./embodied-ai/sim2real.md) — Isaac Lab、Genesis、域随机化、Real-to-Sim
- [具身智能前沿专题](./embodied-ai/frontier.md) — 分层规划、双系统 VLA、Humanoid、灵巧操作

## 模型部署

- [模型部署与服务化](./deployment/model-serving.md) — ONNX、TensorRT、Triton、vLLM
- [模型压缩技术](./deployment/model-compression.md) — 量化 GPTQ/AWQ、剪枝、知识蒸馏

## 资源汇总

- [常用工具命令](./resources/tools.md) — Git、Conda、Docker、Linux
- [常用数据集](./resources/datasets.md)
- [学习资源](./resources/learning-resources.md)
- [大模型求职指南](./resources/llm-career-guide.md) — 岗位地图、技术栈要求、简历/笔试/面试准备与投递策略

# 深度学习笔记 — 文档索引

欢迎来到 deep-learning-notes 文档站。以下是按主题域组织的完整文档索引。

---

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
- [目标检测](./architectures/object-detection.md) — 两阶段与单阶段（Faster R-CNN、YOLO）
- [图像分割](./architectures/segmentation.md) — 语义/实例分割（FCN、U-Net、Mask R-CNN、SAM）
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

- [AI Agent 与 RAG](./agents/ai-agents.md) — ReAct、Function Calling、MCP、检索增强生成

## 具身智能

- [具身智能与 VLA](./embodied-ai/embodied-intelligence.md) — VLA 模型、Diffusion Policy、世界模型、Sim-to-Real

## 模型部署

- [模型部署与服务化](./deployment/model-serving.md) — ONNX、TensorRT、Triton、vLLM
- [模型压缩技术](./deployment/model-compression.md) — 量化 GPTQ/AWQ、剪枝、知识蒸馏

## 资源汇总

- [常用工具命令](./resources/tools.md) — Git、Conda、Docker、Linux
- [常用数据集](./resources/datasets.md)
- [学习资源](./resources/learning-resources.md)

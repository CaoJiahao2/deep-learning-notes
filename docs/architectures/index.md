# 模型架构

> 从 CNN、RNN 到 Transformer，从 GAN、Diffusion 到 Flow Matching —— 架构是模型能力的骨架。

本章节按「主干网络 → 生成模型 → 视觉理解 → 领域应用」四条线组织：主干网络回答「用什么结构做表征」，生成模型回答「如何从噪声中造出数据」，视觉理解回答「如何把表征变成检测框与掩码」，领域应用则覆盖语音、推荐等专门方向。所有架构笔记都遵循同一套分析框架：**动机 → 结构 → 数学 → 复杂度 → 演进脉络**。

## 章节地图

<div class="grid cards" markdown>

-   :material-layers:{ .lg .middle } **主干网络**

    ---

    表征学习的基本盘，读懂一切论文的前置。

    - [Transformer](./transformer.md) — Self-Attention、位置编码，现代模型的公共底座
    - [卷积神经网络 CNN](./cnn.md) — LeNet→ResNet→EfficientNet 演进
    - [循环神经网络 RNN/LSTM/GRU](./rnn-lstm.md) — 序列建模与注意力机制的起源
    - [图神经网络 GNN](./gnn.md) — GCN/GAT/GraphSAGE、消息传递、过平滑
    - [现代架构演进 MoE/SSM](./modern-architectures.md) — 稀疏专家、Mamba、高效注意力

-   :material-creation:{ .lg .middle } **生成模型**

    ---

    从对抗到扩散再到流匹配的范式更替。

    - [生成对抗网络 GAN](./gan.md) — 从 Vanilla GAN 到 StyleGAN
    - [扩散模型 Diffusion](./diffusion.md) — DDPM、Stable Diffusion、Latent Space
    - [Flow Matching 与 Rectified Flow](./flow-matching.md) — SD3、Flux、π0 的统一范式
    - [视频生成](./video-generation.md) — Sora、DiT、3D VAE
    - [可控图像生成](./controllable-image-generation.md) — ControlNet、IP-Adapter
    - [3D 生成：NeRF 与 Gaussian Splatting](./3d-generation.md) — 体渲染、3DGS、4D 场景

-   :material-eye-outline:{ .lg .middle } **视觉理解**

    ---

    把像素变成结构化输出。

    - [目标检测](./object-detection.md) — Faster R-CNN、YOLO 两阶段 vs 单阶段
    - [图像分割](./segmentation.md) — FCN、U-Net、Mask R-CNN、SAM
    - [单目深度与 3D 视觉](./monocular-depth.md) — Depth Anything、Metric3D

-   :material-tune:{ .lg .middle } **领域应用**

    ---

    主干架构在专门领域的变体。

    - [语音与音频大模型](./speech-audio.md) — ASR/TTS、语音 Tokenizer、音乐生成
    - [推荐系统与 CTR 模型](./recommendation.md) — DeepFM/DIN、多目标排序、LLM 推荐

</div>

## 学习建议

1. **先 Transformer 后一切**：[Transformer](./transformer.md) 是 LLM、扩散模型（DiT）、多模态（ViT）与 VLA 的公共底座，务必先吃透。
2. **生成模型按时间线读**：GAN → Diffusion → Flow Matching 是一条清晰的范式演进线，理解「为什么要换范式」比记住公式更重要。
3. **对照复杂度表**：每种架构都建议亲手算一遍时间/空间复杂度，这是面试与设计选型的高频考点。
4. **配套训练知识**：架构只解决「长什么样」，训练技术章节解决「怎么训出来」，见 [训练技术总览](../training/index.md)。

## 延伸阅读

- [现代架构演进 MoE/SSM](./modern-architectures.md) — 不确定下一步读什么时的最佳选择
- [自监督学习](../training/self-supervised-learning.md) — 主干网络如何获得预训练权重
- [LLM/MLLM 全景调研](../llm-mllm/survey.md) — 架构如何堆叠成大模型

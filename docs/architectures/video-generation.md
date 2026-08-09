# 视频生成

> 从图像生成延伸到时序维度。Diffusion Transformer (DiT) 是 2025 主流架构，配套 3D VAE 压缩时空表征。

---

## 目录

1. [任务定义与挑战](#1)
2. [架构演进](#2)
3. [扩散 Transformer (DiT)](#3)
4. [时空压缩：3D VAE 与潜空间](#4)
5. [代表模型](#5)
6. [训练与数据](#6)
7. [条件控制](#7)
8. [评测基准](#8)
9. [工程实践](#9)
10. [参考文献](#10)

---

## 1 任务定义与挑战 {#1}

视频生成是文生图模型的"时间扩展"：

- **文生视频 (T2V)**：给定一段文本提示，生成短视频；
- **图生视频 (I2V)**：给定首帧与文本，生成后续帧；
- **视频续写 (V2V)**：给定已有视频与文本，扩展或编辑；
- **视频超分 / 修复**：把低分辨率/低质量视频升级。

挑战远高于图像：

- **时空一致性**：相邻帧必须连贯，不能"闪烁"或"跳变"；
- **运动合理性**：物体运动应符合物理与人对世界的常识；
- **长程记忆**：长视频中物体 / 角色应保持一致身份；
- **算力爆炸**：视频是图像的 N 倍像素 × T 帧，参数量与显存都剧增。

---

## 2 架构演进 {#2}

视频生成架构经历了三代演进：

```text
2D U-Net（Image Diffusion）
   │
   ▼
3D U-Net（时空同时建模）
   │
   ▼
视频 DiT（Diffusion Transformer）
   │
   ▼
Video DiT + 3D VAE + 时空分离注意力
```

### 2.1 3D U-Net 时代

把 UNet 的 2D 卷积扩展为 3D（含时间维度）。代表：Make-A-Video、AnimateDiff。

### 2.2 DiT 时代

扩散模型 Transformer（DiT）在图像上击败 UNet 后，迅速被视频模型采用。代表：Sora、Wan 2.1、CogVideoX、HunyuanVideo。

### 2.3 时空分离注意力

把"空间注意力"与"时间注意力"分两个阶段计算：

```text
x → spatial-attn → temporal-attn → next block
```

优势：参数量与算力可控，且空间结构可与图像 DiT 权重共享。Wan 2.1、HunyuanVideo 都采用这一设计。

---

## 3 扩散 Transformer (DiT) {#3}

### 3.1 核心思想

把 UNet 替换为 Transformer，所有条件（文本、时间步、类别）通过 adaptive LayerNorm 注入：

$$\text{AdaLN}(h, t) = \gamma(t) \cdot \text{LayerNorm}(h) + \beta(t)$$

其中 $\gamma(t), \beta(t)$ 由时间步 $t$ 经 MLP 预测。

### 3.2 Patch 化

把 $H \times W \times C$ 的特征切成 $p \times p$ 的 patches，序列化为 Transformer 输入 token。

### 3.3 规模化优势

Transformer 在数据规模与算力增加时表现稳健，这就是 DiT 在 SD3、Flux、Sora 等 2025 主流模型中全面取代 UNet 的根本原因。

---

## 4 时空压缩：3D VAE 与潜空间 {#4}

### 4.1 为什么需要压缩

直接在像素空间做视频扩散代价过高（O(H × W × T × C)）。主流做法：先用 3D VAE 把视频压到潜空间，再在潜空间做扩散。

### 4.2 3D VAE 结构

- **编码器**：时空下采样（如 4×8×8 = 256 倍压缩率）；
- **解码器**：时空上采样；
- **潜空间**：$(T/h) \times (H/w) \times (W/w) \times C'$，$C' \in \{4, 16\}$。

Wan 2.1 的 VAE 在 256 倍压缩下仍保持高重建质量，是其能在 14B 模型上生成 5 秒 1080p 的关键。

### 4.3 替代方案

- **时间压缩单独**：仅对空间压缩，时间保持；
- **Token 化**：用 VideoGPT 的 VQ-VAE 进一步离散化；
- **Patch 化**：像图像 DiT 那样直接 patch 时间维度。

---

## 5 代表模型 {#5}

### 5.1 Sora（OpenAI, 2024 / 2025）

- 第一个引起业界广泛关注的高质量视频生成模型；
- 推测采用 Video DiT + 3D VAE；
- 2025 年发布 Sora 2，原生音频生成，60s+ 长视频。

### 5.2 Veo 2（Google DeepMind, 2025）

- 4K 分辨率，强项"电影级"质感；
- 与 Gemini 生态深度集成。

### 5.3 Kling 2.0（Kuaishou）

- 改进运动动力学、角色一致性、1080p；
- 在中国市场份额大。

### 5.4 Wan 2.1（Alibaba, 2025）

- 开源（Apache 2.0）；
- 1.3B 与 14B 两档，14B 在多个基准上超过 Kling / Veo / Sora；
- 消费级 GPU 可运行 1.3B。

### 5.5 CogVideoX（Zhipu）

- 开源中文视频模型；
- 3D VAE + Expert Transformer。

### 5.6 HunyuanVideo（Tencent）

- 13B 参数量，开源；
- 强调长视频与可控性。

### 5.7 Movie Gen（Meta, 2024）

- 30B+ 参数；
- 强调高保真物理与声音同步。

### 5.8 LTX-Video（Lightricks）

- 实时视频生成（在消费级 GPU 上实时）；
- 强调低延迟。

### 5.9 Veo 3 / Kling 3.0（2026 趋势）

- 2026 趋势：更长时长（数分钟）、原生音频、可控镜头。

---

## 6 训练与数据 {#6}

### 6.1 视频数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| Panda-70M | 70M 视频 | 自动流水线生成 |
| Video-CC | 100M+ | 类似图像 CC 的视频版 |
| HD-VG-130M | 130M | 高清 |
| InternVideo | 大 | 学术多任务 |
| WebVid | 10M | 早期学术基线 |

### 6.2 数据预处理

- 镜头检测与分段（避免镜头切换打散时序）；
- 字幕/旁白提取（用 Whisper）；
- 文本-视频对齐（用 vCLIP / VideoCLIP）；
- 质量过滤（美学分数、运动幅度、解码器重建误差）。

### 6.3 训练技巧

- **图像 + 视频联合预训练**：先在图像上学"世界知识"，再在视频上学"运动"；
- **课程学习**：从短视频（2s）到长视频（10s+）；
- **Teacher Distillation**：用大模型做教师，蒸馏到小模型；
- **多阶段训练**：低分辨率 → 高分辨率；潜空间 → 像素空间（可选）。

---

## 7 条件控制 {#7}

### 7.1 文本条件

主流做法：把文本用 T5 或 CLIP 文本编码器编码为 token，与视频潜空间 token 在 cross-attention 中融合。

### 7.2 图像条件（I2V）

把首帧作为附加条件注入：

- **CLIP Image Embedding**：作为全局风格引导；
- **Latent Concatenation**：把首帧潜向量拼到每帧；
- **Reference Attention**：把首帧作为"参考 token"。

### 7.3 镜头控制

- **CameraCtrl**：可调节相机轨迹（平移、推拉、环绕）；
- **MotionCtrl**：用轨迹控制运动幅度；
- **Direct-a-Video**：直接控制相机参数。

### 7.4 音频条件（2025 新趋势）

Sora 2 / Veo 2 引入原生音频生成：

- 在视频扩散的潜空间之上加音频头；
- 文本-音频-视频三者联合训练。

---

## 8 评测基准 {#8}

### 8.1 自动指标

- **FVD**（Fréchet Video Distance）：衡量生成视频分布与真实的差距；
- **IS**（Inception Score）：图像质量的迁移指标；
- **CLIPScore**：文本-视频对齐。

### 8.2 综合基准

- **VBench**：从 16 个维度（人物一致性、运动平滑度、相机、色彩…）打分；
- **ChronoEdit**：时间一致性；
- **EvalCrafter**：任务级评测；
- **FETV**：人类偏好数据。

### 8.3 评测陷阱

- 自动指标与人类偏好相关性弱（特别是运动合理性）；
- 公开 benchmark 已被"刷分"，需关注评测集分布；
- 长视频评测需关注"长程一致性"。

---

## 9 工程实践 {#9}

### 9.1 显存优化

- **3D VAE 分块**：把视频切成时空小块再编码；
- **梯度检查点**：反向传播时丢弃中间激活；
- **FlashAttention**：替代标准 attention；
- **并行策略**：模型并行 + 序列并行 + 数据并行混合。

### 9.2 推理优化

- **CFG 蒸馏**：把 Classifier-Free Guidance 的双次前向合并为单次；
- **ODE 求解器**：用 Heun / Dopri5 替代 Euler；
- **Cache**：时间步之间的特征复用；
- **量化和编译**：INT8/FP8 + TensorRT/TorchCompile。

### 9.3 消费级部署

Wan 2.1 1.3B 在 24GB 显存 GPU 上可生成 5 秒 720p 视频；再低显存可用：

- 启用 fp8 量化（≈ 2× 显存降低）；
- 启用 xformers/flash-attn；
- 切片推理（把视频切成短段生成再拼接）。

---

## 10 参考文献 {#10}

[1] W. Peebles & S. Xie, *Scalable Diffusion Models with Transformers (DiT)*, ICCV, 2023. arXiv:2212.09748.

[2] OpenAI, *Video Generation Models as World Simulators*, 2024. https://openai.com/index/video-generation-models-as-world-simulators.

[3] OpenAI, *Sora 2: A More Capable Video Generation Model*, 2025.

[4] Google DeepMind, *Veo 2 Technical Report*, 2025.

[5] Wan Team, *Wan2.1: Open and Large-Scale Video Generation with Flow Matching*, 2025. arXiv:2503.08317.

[6] Zhipu AI, *CogVideoX: Text-to-Video Diffusion Models with Expert Transformers*, 2024. arXiv:2408.06072.

[7] Tencent, *HunyuanVideo: A Systematic Framework For Large Video Generative Models*, 2024. arXiv:2412.03603.

[8] Meta, *Movie Gen: A Cast of Media Foundation Models*, 2024. arXiv:2410.13720.

[9] U. Singer et al., *Make-A-Video: Text-to-Video Generation without Text-Video Data*, ICLR, 2023.

[10] Y. Guo et al., *AnimateDiff: Training-Free Inference-Time Animation of Diffusion Models*, 2023.

[11] Z. Wang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, 2023.

[12] H. He et al., *CameraCtrl: Camera Control for Text-to-Video Generation*, 2024.
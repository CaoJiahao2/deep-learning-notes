# 端侧 AI 与移动端推理

> 从 llama.cpp 到 NPU 加速，从量化压缩到端云协同 —— 让大模型在手机、PC 和 IoT 设备上运行。

---

## 目录

1. [概述](#1)
2. [端侧 AI 生态](#2-ai)
3. [模型量化与格式](#3)
4. [推理运行时](#4)
5. [硬件加速](#5)
6. [端云协同](#6)
7. [小模型与高效架构](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

端侧 AI（On-Device AI / Edge AI）是指在用户设备（手机、PC、平板、IoT、车载、机器人）上直接运行模型推理，而非依赖云端。2024-2025 年，随着小模型能力快速提升和专用 AI 硬件普及，端侧 AI 成为消费级 AI 的主战场。

核心驱动力：

- **隐私**：敏感数据（聊天、照片、文档）不上传云端。
- **延迟**：本地推理无网络往返，适合实时交互（语音、相机、键盘预测）。
- **成本**：无云推理费用，适合大规模免费功能。
- **可用性**：无网络时仍可工作。

核心挑战：

- 移动端内存通常 8-16GB，模型需压缩至 3-8B 量化后约 2-6GB。
- 算力和功耗远低于数据中心 GPU。
- 不同硬件平台（Apple Neural Engine、Qualcomm Hexagon、MediaTek APU）碎片化。

---

## 2. 端侧 AI 生态

### 2.1 平台概览

| 平台 | 代表硬件 | 推理框架 | 典型模型规模 |
|------|---------|---------|-------------|
| Apple | M-series Mac, A-series iPhone | Core ML, MLX | 7B-70B（Mac）/ 3B（手机） |
| Qualcomm | Snapdragon 8 Gen 3/4 | QNN, Qualcomm AI Engine | 7B-10B |
| MediaTek | Dimensity 9300/9400 | NeuroPilot, APU | 7B-13B |
| Google | Tensor G4, Pixel | Google AI Edge, LiteRT | 3B-7B |
| NVIDIA | Jetson Orin/Thor | TensorRT, CUDA | 7B-70B |
| 通用 | x86/ARM PC | llama.cpp, ONNX Runtime | 1B-70B |

### 2.2 端侧应用场景

- **智能助手**：手机上的对话式 AI（Apple Intelligence、Google Gemini Nano、Samsung Gauss）。
- **文本输入**：键盘智能回复、语法纠错、摘要。
- **照片/相机**：实时图像增强、目标检测、语义搜索。
- **实时翻译**：语音和文字的离线翻译。
- **代码补全**：本地 Copilot（如 Continue.dev + 本地模型）。
- **车载 AI**：语音助手、驾驶辅助、座舱多模态。
- **IoT/机器人**：本地感知、决策和控制。

---

## 3. 模型量化与格式

### 3.1 GGUF 格式

GGUF（GPT-Generated Unified Format）是 llama.cpp 生态的模型格式，专为 CPU 和混合推理优化：

- 支持多种量化精度：Q4_K_M、Q5_K_M、Q8_0、FP16。
- 量化后模型大小约为 FP16 的 25%-50%。
- 内置分词器和元数据，单文件分发。
- 支持 CPU、GPU（CUDA/Metal/Vulkan）和异构推理。

常见量化精度选择：

| 格式 | 比特/参数 | 7B 模型大小 | 质量损失 |
|------|----------|------------|---------|
| Q4_K_M | ~4.5 | ~4.1 GB | 轻微 |
| Q5_K_M | ~5.5 | ~4.8 GB | 极小 |
| Q8_0 | 8 | ~7.2 GB | 几乎无损 |
| FP16 | 16 | ~14 GB | 无 |

### 3.2 其他量化方案

- **GPTQ**：4-bit 权重量化，需校准数据，GPU 推理速度快，适合 NVIDIA 设备。
- **AWQ（Activation-Aware Weight Quantization）**：保护显著权重通道，4-bit 下质量优秀。
- **FP8/FP4（NVFP4）**：NVIDIA Blackwell 原生支持的低比特浮点格式。
- **Block-wise INT4 / INT8**：NPU 友好的量化方式，按块缩放。
- **MLX 量化**：Apple Silicon 原生 4-bit/8-bit 量化格式。

### 3.3 量化质量评估

量化后需评估质量损失：

- ** perplexity**：量化模型在标准文本上的困惑度，与 FP16 基线对比。
- **任务评测**：MMLU、GSM8K 等基准上的分数变化。
- **主观评估**：实际生成质量的人工对比。
- Q4 量化通常导致 1-5% 的性能损失，在 7B+ 模型上影响可接受，在小模型（1B 以下）上更明显。

---

## 4. 推理运行时

### 4.1 llama.cpp

llama.cpp 是最广泛使用的端侧推理框架：

- 纯 C/C++ 实现，无 CUDA 依赖，支持 CPU、Metal、CUDA、Vulkan、SYCL。
- GGUF 格式原生支持。
- 支持对话生成、embedding、 speculative decoding。
- 绑定丰富：llama-cpp-python、text-generation-webui、LangChain 集成。
- 持续优化：Flash Attention、Paged Attention、KV Cache 量化、多 GPU 并行。

### 4.2 Apple MLX

MLX 是 Apple 为 Apple Silicon 设计的机器学习框架：

- 统一内存架构（Unified Memory）：CPU 和 GPU 共享内存，无数据拷贝。
- 原生支持 Swift/Python API。
- 支持 LoRA 微调、量化、分布式训练。
- MLX-Examples 提供 LLM、Stable Diffusion 的端侧实现。
- 在 M-series 芯片上性能优秀，可流畅运行 70B 量化模型。

### 4.3 其他运行时

| 框架 | 特点 |
|------|------|
| ONNX Runtime | 跨平台，支持 CPU/GPU/NPU/移动端 |
| LiteRT（原 TensorFlow Lite） | Google 移动端推理框架 |
| MLC LLM | 基于 TVM 编译，支持多硬件后端 |
| ExecuTorch | PyTorch 官方端侧推理框架 |
| TensorRT-LLM for Jetson | NVIDIA 嵌入式平台 |
| Qualcomm QNN | Snapdragon NPU 原生 SDK |
| MediaPipe | Google 端侧多模态 AI 框架 |

### 4.4 浏览器端推理

- **WebGPU**：浏览器中直接调用 GPU 计算，支持 Transformers.js、WebLLM。
- **WebAssembly (WASM)**：CPU 回退方案，兼容性好但较慢。
- 典型应用：在浏览器中运行 BERT 分类、Whisper 识别、小型 LLM。

---

## 5. 硬件加速

### 5.1 专用 NPU

移动 NPU（Neural Processing Unit）专为低功耗推理设计：

- **Apple Neural Engine**：A/M 系列芯片中的 NPU，通过 Core ML 调度。
- **Qualcomm Hexagon NPU**：支持 INT4/INT8/FP16，微架构针对 Transformer 优化。
- **MediaTek APU**：支持 Transformer 直接加速。
- NPU 的优势是能效比（TOPS/Watt），但生态和通用性弱于 GPU。

### 5.2 GPU 加速

- **Apple Metal**：M-series/Mac 的 GPU 可通过 Metal 直接运行 llama.cpp 和 MLX。
- **CUDA（Jetson）**：NVIDIA 嵌入式平台支持完整 CUDA 生态。
- **Vulkan / OpenCL**：跨平台 GPU 计算，llama.cpp 支持。
- 移动 GPU（Adreno、Mali）也可通过 Vulkan 运行推理。

### 5.3 CPU 优化

- **SIMD 指令**：AVX2/AVX-512（x86）、NEON（ARM）加速矩阵运算。
- **量化内核**：针对 int4/int8 的优化 GEMM 内核。
- **NUMA 感知**：多插槽服务器/大核小核架构下的线程调度。
- 纯 CPU 推理在现代 PC 上 7B Q4 模型可达 10-30 tokens/s。

---

## 6. 端云协同

### 6.1 混合推理架构

端侧和云端各有优势，实际系统常采用混合策略：

- **简单任务本地处理**：短文本补全、摘要、意图识别用端侧小模型。
- **复杂任务上云**：长文档分析、代码生成、多模态推理用云端大模型。
- **隐私敏感数据本地**：涉及个人信息的内容不出设备。
- **按需路由**：路由器模型判断任务复杂度，决定本地或云端。

### 6.2 模型级联

- 端侧小模型先尝试，置信度低时升级到云端大模型。
- 端侧做预处理（意图识别、实体提取、向量检索），云端做生成。
- 云端生成 + 端侧后处理（格式调整、敏感信息过滤）。

### 6.3 端侧个性化

端侧模型可结合用户私有数据进行个性化：

- 在设备上用用户笔记、邮件、聊天记录微调（LoRA）。
- 端侧 RAG：文档和索引存储在本地，检索后送模型。
- 个性化数据不离开设备，保护隐私。

---

## 7. 小模型与高效架构

### 7.1 代表性端侧模型

| 模型 | 规模 | 特点 |
|------|------|------|
| Phi-3 / Phi-4 | 3.8B/14B | 微软"教科书级"训练，小模型强能力 |
| Gemma 2 | 2B/9B | Google 开源，端侧优化 |
| Qwen2.5 | 0.5B-7B | 阿里，多尺寸，中文强 |
| Llama 3.2 | 1B/3B | Meta 端侧模型，多模态 |
| Mistral / Mixtral | 7B (8x7B) | 高效架构，MoE 可选 |
| MiniCPM | 1B-4B | 面壁智能，端侧多模态 |
| DeepSeek | 1.5B-7B | 代码/推理能力强 |
| SmolLM | 135M-1.7B | Apple 端侧小模型研究 |

### 7.2 架构优化

- **Grouped-Query Attention (GQA)**：减少 KV Cache 大小，详见 `llm-mllm/inference.md`。
- **MoE 端侧**：Mixtral 8x7B 总参数 47B 但每 Token 仅激活 13B，内存需求不变。
- **深度可分离卷积 / Mamba 混合**：减少注意力计算，适合端侧。
- **知识蒸馏**：用大模型教小模型，保留大部分能力。

---

## 8. 实践建议

### 8.1 模型选型

- 手机实时交互（< 3B）：Qwen2.5-1.5B/3B、Llama 3.2-3B、Phi-3-mini。
- PC/Mac 流畅使用（7B）：Qwen2.5-7B、DeepSeek-7B、Mistral-7B。
- Mac 高端/工作站（32B+）：Qwen2.5-32B Q4、Llama 3.1-70B Q4。
- 中文场景优先 Qwen/DeepSeek，代码优先 DeepSeek-Coder，多模态考虑 MiniCPM-V。

### 8.2 量化建议

- 优先 Q4_K_M（质量与大小平衡），存储紧张用 Q3_K_M，追求质量用 Q5_K_M。
- 1B 以下小模型建议用 Q8_0 或 FP16，低比特量化损失较大。
- 量化后务必做 perplexity 和实际任务评测，不要只看 benchmark 分数。
- 使用 imatrix（重要性矩阵）量化可显著改善低比特质量。

### 8.3 性能优化

- KV Cache 类型选择：Q8_0（速度/质量平衡）或 Q4_0（极限压缩）。
- 上下文窗口控制在实际需要的大小，避免浪费内存。
- Apple Silicon 上 MLX 通常比 llama.cpp 更快；纯 CPU 场景 llama.cpp 更成熟。
- 使用批处理和连续批处理（continuous batching）提升吞吐。
- 关注首 Token 延迟（TTFT）和 Token 生成速度（TPS）两个指标。

---

## 9. 参考文献

[1] llama.cpp, https://github.com/ggerganov/llama.cpp, 2023-2025.

[2] Apple Inc., *MLX: Machine Learning Framework for Apple Silicon*, 2024.

[3] Microsoft, *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*, 2024.

[4] T. Dettmers, R. Lewis, Y. Belkada, L. Zettlemoyer, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR, 2023.

[5] J. Lin, J. Tang, H. Tang, et al., *AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration*, MLSys, 2024.

[6] S. Yao, M. Mosbach, X. Chen, et al., *MLC LLM: Universal Deployment of Large Language Models*, 2023.

[7] Meta, *Llama 3.2: Enabling the Next Generation of Edge AI*, 2024.

[8] Qwen Team, *Qwen2.5 Technical Report*, 2024.

[9] J. Zhou, P. Gao, X. Zou, et al., *MiniCPM: Unveiling the Potential of Small Language Models on Mobile*, 2024.

[10] P. Warden, S. Ghorbanian, et al., *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*, O'Reilly, 2019.

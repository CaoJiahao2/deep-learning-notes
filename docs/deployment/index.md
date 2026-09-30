# 推理部署

> 模型训得再好，服务的可用性与成本才决定它能不能活下来。

本章节覆盖模型离开训练集群后的完整落地链路：从单次推理的计算细节（Prefill/Decode、KV Cache、Continuous Batching），到服务化与集群部署（TensorRT、Triton、vLLM），到模型压缩（量化、剪枝、蒸馏）与端侧适配（llama.cpp、NPU），最后是应用层的 LLMOps（Prompt 管理、评估流水线、灰度发布）。阅读顺序建议从实现原理到服务化，再到压缩与端侧，最后看工程化。

## 章节地图

<div class="grid cards" markdown>

-   :material-memory:{ .lg .middle } **推理实现**

    ---

    一次生成请求的完整生命周期。

    - [推理实现详解](./inference-implementation.md) — Prefill/Decode、PagedAttention、FlashAttention、PD 解耦

-   :material-server:{ .lg .middle } **服务化**

    ---

    把模型变成稳定可调用的服务。

    - [推理服务化与部署](./model-serving.md) — ONNX、TensorRT、Triton、vLLM

-   :material-package-variant-closed:{ .lg .middle } **压缩与端侧**

    ---

    更小、更快、更省。

    - [模型压缩技术](./model-compression.md) — GPTQ/AWQ 量化、剪枝、蒸馏
    - [端侧 AI 与移动端推理](./edge-on-device.md) — llama.cpp/MLX、GGUF、NPU、端云协同

-   :material-chart-line:{ .lg .middle } **应用工程**

    ---

    让 LLM 应用可运营、可迭代。

    - [LLM 应用工程（LLMOps）](./llm-ops.md) — Prompt 管理、评估流水线、Tracing、灰度发布

</div>

## 学习建议

1. **原理先于框架**：先读 [推理实现详解](./inference-implementation.md) 理解 KV Cache 与调度，再学 vLLM 等框架的配置项会事半功倍——每个参数都对应一个底层机制。
2. **量化要算账**：[模型压缩技术](./model-compression.md) 的每种方法都要权衡精度损失与收益，建议用自己手头的模型做 A/B 实测。
3. **端云协同是趋势**：[端侧 AI](./edge-on-device.md) 与 [个性化](../llm-mllm/personalization.md) 正在合流，值得放在一起看。
4. **工程闭环**：上线不是终点，[LLMOps](./llm-ops.md) 的评估流水线与灰度发布是应用长期存活的关键。

## 延伸阅读

- [推理优化](../llm-mllm/inference.md) — 部署视角之外的算法侧推理加速
- [GPU 架构与 CUDA 编程](../fundamentals/gpu-cuda.md) — 理解推理引擎优化的硬件基础
- [ML 系统可靠性](../fundamentals/reliability.md) — 生产事故与静默回归的应对

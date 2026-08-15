# 推理服务化与部署

> 从模型导出到生产级推理服务的完整指南。

---

## 目录

1. [部署概述](#1)
2. [模型导出格式](#2)
3. [推理引擎](#3)
4. [服务化部署](#4)
5. [性能优化](#5)
6. [监控与运维](#6)
7. [参考文献](#7)

---

## 1. 部署概述

### 1.1 部署场景分类

| 场景 | 延迟要求 | 吞吐要求 | 典型方案 |
|------|---------|---------|---------|
| 在线服务 | < 100ms | 高并发 | vLLM, TensorRT-LLM, Triton |
| 边缘设备 | < 10ms | 低 | ONNX Runtime, CoreML, TFLite |
| 批量推理 | 分钟级 | 高 | Spark + ONNX, Ray |
| 嵌入式 | 实时 | 低 | TFLite Micro, TVM |

### 1.2 部署流程

```text
训练好的模型 → 导出 (TorchScript/ONNX)
    → 优化 (图优化/量化/剪枝)
    → 转换 (TensorRT/CoreML/...)
    → 部署 (REST API/gRPC/嵌入)
    → 监控 (延迟/吞吐/准确率)
```

---

## 2. 模型导出格式

### 2.1 TorchScript

PyTorch 原生序列化格式：

```python
# Trace 方式（推荐）
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")

# Script 方式
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

**限制**：Trace 不支持动态控制流，Script 要求代码符合 TorchScript 语法。

### 2.2 ONNX (Open Neural Network Exchange)

跨框架的模型交换格式：

```python
import torch.onnx

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17
)
```

**优势**：跨框架（PyTorch → TensorRT, ONNX Runtime, OpenVINO）

### 2.3 格式对比

| 格式 | 生态 | 性能 | 灵活性 |
|------|------|------|--------|
| PyTorch eager | 最佳 | 基线 | 最高 |
| TorchScript | PyTorch | 中等 | 受限 |
| ONNX | 跨框架 | 中等 | 中等 |
| TensorRT | NVIDIA | 最高 | 低 |
| CoreML | Apple | 高 | 低 |

---

## 3. 推理引擎

### 3.1 TensorRT

NVIDIA GPU 推理优化引擎：

**优化技术**：
- 层融合（Layer Fusion）：合并 Conv+BN+ReLU 为单个 kernel
- 精度校准：INT8 量化
- 内核自动调优：自动选择最优 kernel 实现
- 动态张量内存：减少显存分配

```python
import tensorrt as trt

# 构建引擎
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse_from_file("model.onnx")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
engine = builder.build_serialized_network(network, config)
```

### 3.2 ONNX Runtime

跨平台推理引擎，支持 CPU/GPU/NPU：

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
output = session.run(None, {"input": input_numpy})
```

### 3.3 OpenVINO

Intel 平台推理优化引擎，支持 CPU/GPU/VPU。

### 3.4 引擎选择

| 场景 | 推荐引擎 |
|------|---------|
| NVIDIA GPU 部署 | TensorRT |
| 跨平台部署 | ONNX Runtime |
| Intel CPU/GPU | OpenVINO |
| Apple 设备 | CoreML |
| 移动端 | TFLite / MNN / NCNN |
| 浏览器 | ONNX Runtime Web |

---

## 4. 服务化部署

### 4.1 Triton Inference Server

NVIDIA 生产级推理服务器：

**特性**：
- 支持多框架（TensorRT, ONNX, PyTorch, TF）
- 动态批处理（Dynamic Batching）
- 模型并发执行（Concurrent Model Execution）
- 多模型流水线（Model Ensemble / BLS）
- GPU 显存共享

**模型配置**：
```protobuf
name: "my_model"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ -1, 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 1000 ]
  }
]
instance_group [
  { count: 2, kind: KIND_GPU }
]
dynamic_batching {}
```

### 4.2 FastAPI 自建服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = load_model("model.pt")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    result: str

@app.post("/predict", response_model=Response)
async def predict(req: Request):
    with torch.no_grad():
        result = model(req.text)
    return Response(result=result)
```

### 4.3 部署方案对比

| 方案 | 吞吐 | 延迟 | 复杂度 | 适用场景 |
|------|------|------|--------|---------|
| FastAPI | 低 | 中 | 低 | 原型验证 |
| BentoML | 中 | 中 | 中 | 中小规模 |
| Triton | 高 | 低 | 高 | 生产环境 |
| Ray Serve | 高 | 低 | 中 | 复杂流水线 |
| vLLM | 极高 | 低 | 中 | LLM 推理 |

---

## 5. 性能优化

### 5.1 批处理策略

- **静态批处理**：固定 batch size 处理
- **动态批处理**：服务器端自动合并请求
- **Continuous Batching**：LLM 推理的流式批处理

### 5.2 模型并发

- **多实例部署**：每个实例独立推理
- **模型并行**：单模型跨 GPU 部署
- **流水线**：预处理 → 推理 → 后处理 流水线化

### 5.3 请求优化

- **请求合并**：合并短时间窗口内的请求
- **请求优先级**：高优先级请求优先处理
- **请求缓存**：缓存相同请求的结果

### 5.4 资源规划

| 指标 | 如何估算 |
|------|---------|
| QPS | 压测单个模型实例 |
| 延迟 | P50, P95, P99 延迟 |
| 显存 | 模型大小 + KV Cache + 运行时开销 |
| 实例数 | QPS / 单实例 QPS |

---

## 6. 监控与运维

### 6.1 关键指标

| 指标 | 说明 |
|------|------|
| 请求延迟 | P50/P95/P99 延迟 |
| 吞吐量 | 每秒请求数 (QPS/RPS) |
| 错误率 | 4xx/5xx 比例 |
| GPU 利用率 | SM 利用率、显存带宽 |
| 队列长度 | 等待处理的请求数 |
| 模型漂移 | 输入分布变化 |

### 6.2 监控工具

- **Prometheus + Grafana**：通用指标收集和可视化
- **NVIDIA DCGM**：GPU 指标监控
- **Triton Metrics**：内置推理服务指标

### 6.3 版本管理

```text
model-repository/
├── model_a/
│   ├── 1/          # 版本 1
│   ├── 2/          # 版本 2
│   └── config.pbtxt
└── model_b/
    ├── 1/
    └── config.pbtxt
```

---

## 7. 参考文献

[1] NVIDIA TensorRT Documentation. https://developer.nvidia.com/tensorrt

[2] ONNX Runtime. https://onnxruntime.ai

[3] Triton Inference Server. https://github.com/triton-inference-server/server

[4] Kwon, W., et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023. (vLLM)

[5] BentoML. https://github.com/bentoml/BentoML

[6] Olston, C., et al. TensorFlow-Serving: Flexible, High-Performance ML Serving. arXiv:1712.06139, 2017.

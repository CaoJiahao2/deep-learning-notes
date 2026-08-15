# GPU 架构与 CUDA 编程

> 从 GPU 硬件结构到 CUDA 线程模型，从显存带宽到 Kernel 优化 —— 理解为什么 GPU 适合深度学习、如何写出高效代码。

---

## 目录

1. [概述](#1)
2. [为什么深度学习需要 GPU](#2)
3. [GPU 硬件架构](#3)
4. [CUDA 线程模型](#4)
5. [显存层次与带宽](#5)
6. [CUDA 编程入门](#6)
7. [性能分析与优化](#7)
8. [GPU 生态与 AI 框架](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

GPU（Graphics Processing Unit）最初用于图形渲染，因其**大规模并行**特性被深度学习广泛采用。CUDA（Compute Unified Device Architecture）是 NVIDIA 提供的并行计算平台与编程模型，让开发者能利用 GPU 做通用计算（GPGPU）。

核心概念：

```text
GPU = 成千上万个小核心并行计算
CUDA = 把这些核心组织起来编程的接口
```

对 AI 工程师而言，理解 GPU/CUDA 有助于：选型硬件、预估性能、定位瓶颈、编写/调用高效 kernel、做多 GPU 分布式训练。

---

## 2 为什么深度学习需要 GPU {#2}

### 2.1 CPU vs GPU 设计哲学

| 维度 | CPU | GPU |
|------|-----|-----|
| 核心数 | 少量（几核~几十核） | 数千~数万 |
| 单核性能 | 强（复杂控制逻辑） | 较弱（简单计算单元） |
| 擅长 | 串行、复杂控制流 | 大规模并行简单运算 |
| 内存 | 大容量 DRAM | 高速高带宽 HBM |
| 功耗 | 相对低 | 高 |

### 2.2 深度学习的计算特征

神经网络核心操作是**矩阵乘法**与**卷积**——高度并行、数据规整的计算：

$$Y = XW + b$$

每个输出元素相互独立，天然适合 GPU 并行。GPU 通过 SIMT（单指令多线程）模式让成百上千线程同时执行相同指令处理不同数据。

---

## 3 GPU 硬件架构 {#3}

### 3.1 现代 GPU 结构

以 NVIDIA GPU 为例：

```text
GPU
├── SM (Streaming Multiprocessor) × N
│   ├── CUDA Cores（FP32 单元）
│   ├── Tensor Cores（矩阵加速）
│   ├── Shared Memory（片上共享内存）
│   ├── Registers（寄存器文件）
│   └── 调度器
├── L2 Cache
└── HBM（高带宽显存）
```

- **SM**：GPU 的基本计算单元，含多个 CUDA Core 与 Tensor Core；
- **CUDA Core**：标量运算单元（FP32/INT32 等）；
- **Tensor Core**：专为矩阵乘/卷积设计，支持 FP16/FP8/BF16 混合精度，大幅提升训练与推理性能；
- **HBM**：高带宽显存（如 H100 的 HBM3），是训练/推理数据搬运的关键。

### 3.2 代表性 GPU 对比

| GPU | 显存 | FP32 | FP16 (Tensor) | NVLink | 适用 |
|-----|------|------|---------------|--------|------|
| RTX 4090 | 24GB | 82 TFLOPS | 330 TFLOPS | - | 个人开发 |
| A100 | 80GB | 19.5 TFLOPS | 312 TFLOPS | 600 GB/s | 大模型 |
| H100 | 80GB | 67 TFLOPS | 990 TFLOPS | 900 GB/s | 生产训练 |
| H200 | 141GB | 67 TFLOPS | 990 TFLOPS | 900 GB/s | 长上下文 |

---

## 4 CUDA 线程模型 {#4}

### 4.1 层级结构

CUDA 把线程组织为三级层次：

```text
Grid（网格）
└── Block（线程块）× M
    └── Thread（线程）× N
```

```c
// 内核启动语法
kernel<<<gridDim, blockDim>>>(args);
```

### 4.2 线程标识

线程用内置变量定位：

```c
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx`：线程块在网格中的索引；
- `blockDim`：块内线程数；
- `threadIdx`：线程在块内的索引。

### 4.3 向量加法示例

```c
__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // ... 分配显存、拷贝数据 ...
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    // ... 拷贝回主机 ...
}
```

---

## 5 显存层次与带宽 {#5}

### 5.1 GPU 存储层次

```text
Registers（最快/最小）
→ Shared Memory（片上，块内共享）
→ L2 Cache
→ Global Memory（显存 HBM）
→ Host Memory（CPU 内存，经 PCIe/NVLink）
```

### 5.2 为什么显存带宽重要

深度学习（尤其 Decode 推理）常是 **memory-bound**：

$$t_{\text{compute}} = \frac{\text{FLOPs}}{\text{FLOPS}}, \quad t_{\text{memory}} = \frac{\text{Bytes}}{\text{Bandwidth}}$$

算术强度（Arithmetic Intensity）：

$$AI = \frac{\text{FLOPs}}{\text{Bytes moved}}$$

- $AI$ 高 → compute-bound；
- $AI$ 低 → memory-bound。

H100 的 HBM 带宽约 3.35 TB/s，CPU 内存带宽远低于此；主机↔设备数据传输（PCIe ~几十 GB/s）是常见瓶颈。

### 5.3 内存合并访问

相邻线程访问相邻内存地址（Coalesced Access）时，一次内存事务可服务多个线程，带宽利用率最高。优化 kernel 时优先保证合并访问。

---

## 6 CUDA 编程入门 {#6}

### 6.1 基本流程

1. 分配主机内存并初始化数据；
2. `cudaMalloc` 分配设备（GPU）内存；
3. `cudaMemcpy` 把数据拷贝到设备；
4. 启动 kernel（`<<<grid, block>>>`）；
5. `cudaMemcpy` 把结果拷贝回主机；
6. 释放内存（`cudaFree`）。

### 6.2 常用 API

```c
cudaMalloc(&d_ptr, size);          // 分配显存
cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();            // 同步设备
cudaGetLastError();                 // 检查错误
cudaEventRecord();                  // 计时
```

### 6.3 共享内存与同步

块内线程通过共享内存（`__shared__`）协作，用 `__syncthreads()` 同步：

```c
__shared__ float sdata[256];
sdata[tid] = input[tid];
__syncthreads();
```

共享内存比全局内存快一个数量级，是优化矩阵乘、归约等操作的关键。

### 6.4 异步与流（Stream）

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<blocks, threads, 0, stream>>>(...);   // 异步执行
cudaMemcpyAsync(..., stream);                  // 异步拷贝
```

利用多个 Stream 可实现**计算与数据搬运重叠**，隐藏传输延迟。

---

## 7 性能分析与优化 {#7}

### 7.1 分析工具

- **nvidia-smi**：查看 GPU 利用、显存、功耗；
- **Nsight Systems / Nsight Compute**：性能分析，定位瓶颈；
- **nvprof / Nsight Profiler**：kernel 耗时统计；
- **TensorBoard / PyTorch Profiler**：框架层性能分析。

### 7.2 常见优化手段

| 手段 | 作用 |
|------|------|
| Kernel Fusion | 合并多个 kernel，减少内存搬运与启动开销 |
| 共享内存复用 | 减少全局内存访问 |
| 合并访问 | 提升带宽利用率 |
| 向量化（float4） | 一次搬运更多数据 |
| 展开循环 | 减少循环开销 |
| 异步拷贝 + Stream | 隐藏传输延迟 |
| 混合精度（FP16/FP8） | 提升 Tensor Core 利用率 |
| 减少同步 | 避免不必要的 `__syncthreads` |

### 7.3 估算性能上限

对训练/推理任务，可用 Roofline 模型估算：

$$\text{Peak} = \min\left(\frac{\text{FLOPs}}{t}, \text{Bandwidth}\right)$$

实际性能通常远低于理论峰值，需结合分析与优化逼近。

---

## 8 GPU 生态与 AI 框架 {#8}

### 8.1 CUDA 工具链

```text
CUDA Toolkit（编译器 nvcc、运行时、库）
+ cuBLAS（BLAS 矩阵库）
+ cuDNN（深度学习原语）
+ NCCL（多 GPU 通信）
+ cuFFT / cuSPARSE
```

### 8.2 在 PyTorch 中使用

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)

torch.cuda.is_available()   # 是否可用
torch.cuda.get_device_name(0)
torch.cuda.mem_get_info()   # 显存信息
torch.cuda.empty_cache()    # 清空缓存
```

### 8.3 环境变量

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py   # 指定可见 GPU
CUDA_LAUNCH_BLOCKING=1 python debug.py     # 同步定位错误
NCCL_DEBUG=INFO python dist.py             # 调试多卡通信
```

### 8.4 多 GPU 与异构

- 多 GPU 训练用 NCCL + DDP（见 `distributed.md`）；
- GPU 集群/基础设施见 `ai-infrastructure.md`；
- 端侧/非 NVIDIA 平台（Apple MLX、NPU）见 `edge-on-device.md`。

---

## 9 参考文献 {#9}

[1] NVIDIA, *CUDA C++ Programming Guide*, NVIDIA Docs.

[2] J. Nickolls, I. Buck, M. Garland, K. Skadron, *Scalable Parallel Programming with CUDA*, ACM Queue, 2008.

[3] D. B. Kirk, W.-M. W. Hwu, *Programming Massively Parallel Processors*, Morgan Kaufmann.

[4] NVIDIA, *H100 Tensor Core GPU Architecture*, 2022.

[5] NVIDIA, *cuBLAS / cuDNN / NCCL Documentation*.

[6] A. Paszke, S. Gross, F. Massa, et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS, 2019.

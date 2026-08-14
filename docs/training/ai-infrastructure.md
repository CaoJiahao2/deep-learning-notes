# AI 训练基础设施

> 从 GPU 集群网络到容错调度，从存储栈到成本模型 —— 万卡集群的工程基石。

---

## 目录

1. [概述](#1)
2. [GPU 与计算节点](#2-gpu)
3. [高速网络与集合通信](#3)
4. [存储栈](#4)
5. [训练框架与并行策略](#5)
6. [调度与容错](#6)
7. [Checkpoint 与恢复](#7-checkpoint)
8. [成本模型与容量规划](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1. 概述

训练千亿参数级大模型需要数千至数万张 GPU 协同工作，这对基础设施提出了极高要求。AI 训练基础设施涵盖硬件选型、网络拓扑、存储 I/O、训练框架、作业调度和故障恢复等多个层面，是大模型训练成功的工程基石。

与分布式训练算法（见 `training/distributed.md`）关注"如何切分模型和数据"不同，本篇关注"如何让数万张 GPU 高效稳定地一起运行"——这是系统工程问题。

核心挑战：

- **通信瓶颈**：算力增长快于网络带宽增长，AllReduce 通信常成为瓶颈。
- **故障率**：万卡集群平均每小时至少有一次硬件故障，训练必须能容错恢复。
- **I/O 瓶颈**：加载数 TB 数据和保存检查点需要高吞吐存储。
- **成本**：大规模训练成本达数百万至数千万美元，需精细规划。

---

## 2. GPU 与计算节点

### 2.1 GPU 选型

| GPU | 显存 (HBM) | BF16 算力 | NVLink 带宽 | 典型用途 |
|-----|-----------|----------|------------|---------|
| NVIDIA A100 | 40/80 GB | 312 TFLOPS | 600 GB/s | 前代主力 |
| NVIDIA H100 | 80 GB | 990 TFLOPS | 900 GB/s | 当前主力 |
| NVIDIA H200 | 141 GB | 990 TFLOPS | 900 GB/s | 长上下文/大 MoE |
| NVIDIA B200 | 192 GB | 2250 TFLOPS | 1.8 TB/s | 下一代 |
| AMD MI300X | 192 GB | 1300 TFLOPS | Infinity Fabric | 替代方案 |
| Google TPU v5p | 95 GB | 459 TFLOPS | ICI | Google 生态 |

### 2.2 节点内拓扑

每个计算节点通常包含 4-8 张 GPU，通过高速总线互联：

- **NVLink/NVSwitch**：GPU 间的高带宽直连。H100 SXM 通过 NVSwitch 实现全互联拓扑，8 GPU 间总带宽 900 GB/s。
- **PCIe**：非 SXM GPU（如 PCIe 版 H100）通过 PCIe Gen5 连接，带宽低于 NVLink。
- **NUMA 亲和性**：GPU 与 CPU socket 的物理连接影响 CPU↔GPU 传输速度，需正确绑定。
- **多实例 GPU（MIG）**：A100/H100 支持切分为多个小实例，适合推理和小任务。

节点间通过 RDMA 网络（InfiniBand 或 RoCE）连接。

---

## 3. 高速网络与集合通信

### 3.1 网络架构

大模型训练集群通常采用 Fat-Tree（胖树）或 Rail-Optimized 拓扑：

- **Fat-Tree**：多层交换机构建无阻塞网络，所有节点间带宽相同。
- **Rail-Optimized**：每张 GPU 的网卡连接到不同的"轨道"（rail），同 rail 内的 GPU 无需跨交换机通信。这降低了交换机层级和延迟。
- **后端网络 vs 前端网络**：训练数据网络（后端）和管理网络（前端）物理隔离。

典型 H100 集群配置：每个节点 8 张 GPU，每张 GPU 配一块 400Gbps ConnectX-7 NIC，8 条上行链路到 Rail 交换机。

### 3.2 RDMA

RDMA（Remote Direct Memory Access）允许 GPU 直接访问其他节点的显存，绕过 CPU 和内核网络栈：

- **InfiniBand（IB）**：原生 RDMA，延迟低（~1μs），带宽高（400Gbps NDR），是 HPC/AI 集群首选。
- **RoCEv2（RDMA over Converged Ethernet）**：在以太网上实现 RDMA，成本较低但需无损网络配置（PFC/ECN）。
- **GPUDirect RDMA**：GPU 显存直接通过 NIC 发送，不经系统内存，延迟降低 50%+。

### 3.3 集合通信

NCCL（NVIDIA Collective Communications Library）是 GPU 集合通信的事实标准：

- **AllReduce**：数据并行中同步梯度。Ring AllReduce 带宽利用率最优，Tree AllReduce 在大规模下延迟更优。
- **AllGather / ReduceScatter**：张量并行中收集和分发参数切片。
- **Send/Recv**：流水线并行中的点对点通信。

NCCL 自动检测拓扑（NVLink、PCIe、IB）并选择最优算法和通道。配置参数：

```bash
export NCCL_IB_DISABLE=0          # 启用 InfiniBand
export NCCL_NET_GDR_LEVEL=5       # 启用 GPUDirect RDMA
export NCCL_DEBUG=INFO            # 调试时查看通信详情
export NCCL_P2P_LEVEL=NVL         # 优先使用 NVLink P2P
```

### 3.4 通信优化

- **计算-通信重叠**：反向传播中，梯度就绪即发起异步 AllReduce，同时计算下一层梯度。
- **梯度分桶（Gradient Bucketing）**：将多个小张量打包成大 AllReduce 操作，利用带宽。
- **压缩通信**：梯度量化（FP16/BF16/FP8）减少传输量，误差累积时用误差补偿。
- **拓扑感知调度**：将通信量大的并行组（如张量并行）放在同一节点或同一 rail。

---

## 4. 存储栈

### 4.1 训练数据存储

大模型训练数据集通常达数 TB 至数十 TB，需要高吞吐并行读取：

- **并行文件系统**：Lustre、GPFS（IBM Spectrum Scale）、BeeGFS，提供高聚合带宽。
- **分布式对象存储**：S3 兼容存储（MinIO、Ceph），通过 Alluxio 等缓存层加速。
- **NVMe 本地缓存**：训练节点上的高速 SSD 缓存热数据，减少网络存储压力。
- **数据预处理**：将原始数据预先 Tokenize 并保存为二进制格式（如 WebDataset、Mosaic MDS），避免训练时 CPU 瓶颈。

### 4.2 Checkpoint 存储

模型检查点文件巨大（70B 模型 BF16 约 140GB，优化器状态 2x 或更多），需要：

- 高写入带宽：数千张 GPU 同时保存 checkpoint 时达到数百 GB/s。
- 异步 checkpoint：在后台写入，不阻塞训练（见第 7 节）。
- 分层存储：最新 checkpoint 存在高速 NVMe，历史 checkpoint 迁移到对象存储。

### 4.3 数据加载优化

- **DataLoader 多进程预取**：CPU worker 提前加载和预处理 batch。
- **数据分片（Sharding）**：每个 GPU 读取数据的一个子集，避免重复读取。
- **流式加载**：对于超大数据集，流式读取而非全量下载。
- **缓存策略**：热点数据缓存到本地 NVMe，减少重复网络 I/O。

---

## 5. 训练框架与并行策略

### 5.1 主流框架对比

| 框架 | 开发者 | 特点 |
|------|--------|------|
| Megatron-LM | NVIDIA | 张量/流水线/序列并行的标杆实现 |
| DeepSpeed | Microsoft | ZeRO 优化器分片、3D 并行、易用性 |
| FSDP | PyTorch 原生 | 完全分片数据并行，集成 PyTorch |
| Colossal-AI | HPC-AI Tech | 多维并行、异构内存 |
| NeMo | NVIDIA | 端到端训练框架，集成 Megatron |
| MaxText | Google | TPU/JAX 上的高性能 LLM 训练 |

并行策略的详细讨论见 `training/distributed.md`。

### 5.2 框架选择建议

- **GPU 集群、追求极致性能**：Megatron-LM 或 NeMo。
- **快速实验、易用性优先**：DeepSpeed 或 FSDP。
- **TPU 集群**：MaxText/JAX。
- **超大模型（>100B）**：Megatron + DeepSpeed/FSDP 组合 3D 并行。
- **MoE 模型**：Megatron 的 MoE 支持或自研框架（DeepSeek 等）。

---

## 6. 调度与容错

### 6.1 集群调度器

- **Kubernetes + Kubeflow / Volcano**：云原生调度，支持 GPU 拓扑感知、gang scheduling。
- **Slurm**：传统 HPC 调度器，在 AI 集群中仍广泛使用。
- **自研调度器**：大厂（Meta、字节）多自研调度器，优化拓扑感知和弹性调度。

### 6.2 Gang Scheduling

分布式训练需要所有 Worker 同时启动，缺一不可。Gang scheduling 确保作业要么获得全部所需资源，要么排队等待，避免部分资源占用导致死锁。

### 6.3 故障类型与处理

| 故障类型 | 频率 | 影响 | 处理方式 |
|---------|------|------|---------|
| GPU 不可修正错误（Xid） | 高 | 单节点崩溃 | 自动剔除节点，从 checkpoint 恢复 |
| 网络故障 | 中 | 集合通信超时 | NCCL 超时重试，失败则重启 |
| 节点宕机 | 中 | 丢失 8 个 GPU | 替换节点，恢复 checkpoint |
| 存储故障 | 低 | 数据/检查点丢失 | 副本和纠删码 |
| 软件 Bug | 低 | 训练挂起或 NaN | 监控指标自动报警和回滚 |

### 6.4 容错策略

- **心跳检测**：每个 Worker 定期向调度器发送心跳，超时判定故障。
- **NCCL 超时**：设置 `NCCL_TIMEOUT`，通信超时后 abort 整个作业。
- **弹性训练（Elastic Training）**：支持在训练过程中动态增减节点，TorchElastic 是参考实现。
- **自动重调度**：故障后调度器自动申请替换节点，从最新 checkpoint 恢复。
- **NaN 检测**：监控 loss 和梯度，出现 NaN 时自动回滚到最近有效 checkpoint 并降低学习率。

---

## 7. Checkpoint 与恢复

### 7.1 Checkpoint 内容

完整 checkpoint 需保存：

- 模型权重（分片存储）。
- 优化器状态（Adam 的 momentum 和 variance，通常是模型大小的 2 倍）。
- 学习率调度器状态。
- 数据加载器位置（epoch、seed、已读样本偏移）。
- 训练进度（global step、loss history）。

70B 模型 BF16 的 checkpoint 约 140GB 权重 + 280GB 优化器状态 + 其他 ≈ 450GB+。

### 7.2 异步 Checkpoint

同步保存 checkpoint 会暂停训练数十秒到数分钟。异步 checkpoint 在 GPU 内存中将状态拷贝到 CPU 内存，后台线程写入存储：

```text
GPU 状态 → DMA 到 CPU 内存 → 立即恢复训练 → 后台写入分布式存储
```

DeepSpeed、Megatron 均支持异步 checkpoint。

### 7.3 检查点策略

- **保存频率**：通常每 100-1000 步保存一次，平衡恢复时间和存储成本。
- **保留数量**：保留最近 3-5 个 checkpoint，旧 checkpoint 删除或归档。
- **增量 checkpoint**：只保存与上一版本的差异，减少写入量。
- **权重分片**：每个 GPU 只保存自己的分片，避免单节点写入瓶颈。
- **恢复优化**：从最新 checkpoint 恢复，同时检查数据加载位置确保不重复或丢失样本。

---

## 8. 成本模型与容量规划

### 8.1 训练成本估算

训练成本主要由 GPU 小时决定：

$$
\text{成本} = \frac{6 \times N \times D}{\text{MFU} \times \text{GPU 有效算力}} \times \text{每 GPU 小时单价}
$$

其中 $N$ 是参数量，$D$ 是训练 Token 数，6 来自 Transformer 每 Token 约 6 FLOPs/参数的经验估计。MFU（Model FLOPs Utilization）通常在 30%-55%。

示例：训练 70B 模型、2T Token，在 H100 上 MFU 45%：

$$
\text{GPU 小时} = \frac{6 \times 70 \times 10^9 \times 2 \times 10^{12}}{0.45 \times 990 \times 10^{12}} \approx 1.9 \times 10^6 \text{ GPU 小时}
$$

约合 256 张 H100 训练约 1 年，或 2048 张 H100 训练约 40 天。

### 8.2 成本优化

- **提高 MFU**：优化并行策略、通信重叠、数据加载，将 MFU 从 30% 提升到 50% 相当于节省 40% 成本。
- **Spot/Preemptible 实例**：云场景使用竞价实例可节省 50%-70% 成本，但需配合容错。
- **混合精度**：BF16/FP8 训练加速并减少显存。
- **模型架构优化**：MoE、GQA、FlashAttention 降低每 Token 计算量。
- **数据质量**：高质量数据减少达到目标性能所需的 Token 数。

### 8.3 容量规划

- **GPU 数量**：根据目标训练时间和模型规模计算。
- **网络**：确保无阻塞带宽，网络瓶颈会直接拉低 MFU。
- **存储**：至少提供 2-3x 模型大小的检查点空间，以及数据集和日志存储。
- **故障率预算**：预留 5%-15% 的冗余节点替换故障硬件。
- **开发/实验集群**：除生产训练集群外，预留小规模调试集群。

---

## 9. 实践建议

### 9.1 新集群上线检查清单

- [ ] GPU  burn-in 测试：运行 24-72 小时压力测试，淘汰早期故障硬件。
- [ ] 网络验证：运行 NCCL all-reduce 测试，确认跨节点带宽和延迟达标。
- [ ] 存储 I/O 测试：测量聚合读写带宽，确保数据加载和 checkpoint 不拖慢训练。
- [ ] 拓扑检查：`nvidia-smi topo -m` 确认 NVLink/PCIe/NIC 拓扑正确。
- [ ] 容错演练：手动 kill 节点，验证自动恢复流程。

### 9.2 训练运维建议

- 监控 GPU 利用率、温度、功耗、显存，识别异常节点。
- 监控 NCCL 通信时间占比，超过 30% 需排查网络或并行配置。
- 维护硬件黑名单：频繁故障的 GPU/节点及时报修。
- 每次训练记录完整配置（模型、数据、超参、代码版本、硬件环境），确保可复现。
- 建立小规模复现通道：在 8-32 GPU 上验证训练流程后再提交大规模作业。

### 9.3 与其他主题的衔接

- 并行算法（DP/TP/PP/ZeRO）详见 `training/distributed.md`。
- 训练流程和超参数规划详见 `training/training-pipeline.md`。
- 推理服务和部署详见 `deployment/model-serving.md`。
- 数据工程和语料处理详见 `llm-mllm/data-engineering.md`。

---

## 10. 参考文献

[1] D. Narayanan, M. Shoeybi, J. Casper, et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC, 2021.

[2] J. Rasley, S. Rajbhandari, O. Ruwase, Y. He, *DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters*, KDD, 2020.

[3] A. Gholami, Z. Yao, S. Kim, et al., *AI and Memory Wall*, IEEE Micro, 2024.

[4] A. Sergeev, M. Del Balso, *Horovod: Fast and Easy Distributed Deep Learning in TensorFlow*, arXiv preprint, 2018.

[5] PyTorch Team, *FSDP: Fully Sharded Data Parallel*, PyTorch Documentation, 2022.

[6] M. Li, D. G. Andersen, J. W. Park, et al., *Scaling Distributed Machine Learning with the Parameter Server*, OSDI, 2014.

[7] S. Rajbhandari, J. Rasley, O. Ruwase, Y. He, *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, SC, 2020.

[8] NVIDIA, *NCCL: Optimized Primitives for Collective Multi-GPU Communication*, 2024.

[9] T. Dao, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS, 2022.

[10] Q. A. Chen, Z. Bai, S. Gao, et al., *Mind Benchmarks: Evaluating Efficiency of AI Training Clusters*, arXiv preprint, 2024.

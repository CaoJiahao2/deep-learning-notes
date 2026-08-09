# 分布式训练详解

> 从单卡到千卡集群，大规模深度学习训练的完整技术栈。

---

## 目录

1. [分布式训练概述](#1)
2. [数据并行 (DP/DDP)](#2-dpddp)
3. [模型并行](#3)
4. [混合并行与 3D 并行](#4-3d)
5. [ZeRO 优化 (DeepSpeed)](#5-zero-deepspeed)
6. [通信原语](#6)
7. [实战配置](#7)
8. [参考文献](#8)

---

## 1. 分布式训练概述

### 1.1 为什么需要分布式训练

- **模型太大**：单卡显存放不下（如 GPT-3 175B 需要 ~350GB 显存）
- **数据太多**：单卡训练时间过长
- **加速收敛**：通过大 batch size 和更多计算资源加速训练

### 1.2 并行策略分类

| 策略 | 核心思想 | 适用场景 |
|------|---------|---------|
| 数据并行 (DP) | 每卡持有完整模型，分配不同数据 | 模型能放入单卡 |
| 张量并行 (TP) | 将单层权重切分到多卡 | 单层参数过大 |
| 流水线并行 (PP) | 将不同层分配到不同卡 | 层数过多 |
| 序列并行 (SP) | 将长序列切分到多卡 | 极长序列 |
| 专家并行 (EP) | MoE 专家分布到不同卡 | MoE 模型 |

---

## 2. 数据并行 (DP/DDP)

### 2.1 DataParallel (DP)

```python
model = nn.DataParallel(model)
output = model(input)
```

**工作原理**：
1. 将 batch 切分到各 GPU
2. 每张卡独立前向传播
3. 在主卡上聚合梯度
4. 更新参数后广播到所有卡

**缺点**：
- 主卡负载不均（GPU 0 承担聚合和广播）
- Python GIL 限制
- 单进程多线程，效率低

### 2.2 DistributedDataParallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

**工作原理**：
1. 每张卡一个独立进程
2. 通过 AllReduce 同步梯度
3. 各卡独立更新参数（参数始终一致）

**DDP 优势**：
- 多进程，无 GIL 限制
- 通信与计算重叠（通过 gradient bucketing）
- 近线性加速比

### 2.3 梯度累积模拟大 Batch

当 8 卡 × batch_size=32 仍不够时：

```python
# 原始 batch_size = 32, 8 卡 = 256
# 目标 batch_size = 1024, 需要 4 步梯度累积

for i, batch in enumerate(train_loader):
    loss = criterion(model(batch), labels) / 4
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 3. 模型并行

### 3.1 张量并行 (Tensor Parallelism)

将单层权重矩阵沿行或列切分到多张 GPU。

**行并行（Row Parallel）**：
$$Y = XA, \quad X \in \mathbb{R}^{b \times d}, A \in \mathbb{R}^{d \times h}$$

将 $A$ 沿列切分：$A = [A_1 | A_2]$，每张 GPU 计算 $Y_i = XA_i$，最后拼接。

**列并行（Column Parallel）**：
$$Y = XB, \quad X \in \mathbb{R}^{b \times h}, B \in \mathbb{R}^{h \times d}$$

将 $B$ 沿行切分，每张 GPU 先计算 $Y_i = X_i B_i$，再通过 AllReduce 求和。

**Megatron-LM 的 TP 方案**：

```text
Transformer 层中的张量并行：
- Self-Attention: 按头切分 Q, K, V 权重
- FFN: 沿列切分 h→4h，沿行切分 4h→h
- 每层需要 2 次 AllReduce
```

### 3.2 流水线并行 (Pipeline Parallelism)

将不同层分配到不同 GPU，形成流水线。

**GPipe 方案**：
- 将 micro-batch 进一步切分
- 流水线执行，减少 GPU 空闲时间（bubble）

```text
GPU 0: [Layer 1-2] → [Layer 1-2] → [Layer 1-2]  (bubble)
GPU 1: (bubble) → [Layer 3-4] → [Layer 3-4] → [Layer 3-4]
GPU 2: (bubble)  (bubble) → [Layer 5-6] → [Layer 5-6] → [Layer 5-6]
```

**1F1B（One-Forward-One-Backward）调度**：
- 交替执行前向和反向
- 减少显存峰值（不需要保存所有 micro-batch 的激活值）

---

## 4. 混合并行与 3D 并行

### 4.1 3D 并行 (TP + PP + DP)

Megatron-LM 提出的三维并行策略：

```text
    数据并行 (DP)
    ┌──────────────────────────────┐
    │  GPU 0  GPU 1 │ GPU 2  GPU 3 │  ← 张量并行 (TP)
    │  GPU 4  GPU 5 │ GPU 6  GPU 7 │
    └──────────────────────────────┘
    流水线并行 (PP)
```

### 4.2 3D 并行 + ZeRO

现代大模型训练的标准方案：

- Megatron-LM：TP + PP
- DeepSpeed ZeRO：DP 优化
- 两者结合：Megatron-DeepSpeed 集成

---

### 4.3 前沿并行策略（2024–2025）

- **FSDP / FSDP2**：PyTorch 官方"分片数据并行"，将模型参数/梯度/优化器状态分片到各 GPU，可按需 AllGather；FSDP2 更进一步，将参数分片从"整层"细化到"参数分块"，与 TP/序列并行更易组合。
- **Ring Attention**：将 KV 分块到多设备并循环传递，注意力计算与通信重叠，可在有限显存下支持**百万 token 级**上下文。
- **序列并行（Sequence Parallelism）**：把长序列沿序列维度切分，配合 Megatron 的 LayerNorm/Dropout 分片，缓解长序列显存瓶颈。
- **MoE 专家并行（Expert Parallel）**：MoE 模型把不同专家放到不同 GPU，通过 All-to-All 通信路由 token 到对应专家，是 DeepSeek / Mixtral 等大规模 MoE 训练的标准做法。

---

## 5. ZeRO 优化 (DeepSpeed)

### 5.1 显存占用分析

训练时显存占用 = 模型状态 + 剩余状态：

| 组件 | 大小 (字节) | 说明 |
|------|------------|------|
| 参数 (Parameters) | $2\Psi$ | FP16 参数 |
| 梯度 (Gradients) | $2\Psi$ | FP16 梯度 |
| 优化器状态 (Optimizer States) | $12\Psi$ | FP32 参数 + 动量 + 方差 |
| 激活值 (Activations) | 取决于 batch 和层数 | 可通过 checkpointing 折半 |

以 7.5B 参数模型、Adam 优化器为例：
- 参数：15 GB
- 梯度：15 GB
- 优化器状态：90 GB
- 总计：120 GB > 单卡 80GB A100

### 5.2 ZeRO 三个阶段

| 阶段 | 分片内容 | 显存节省 | 通信量 |
|------|---------|---------|--------|
| ZeRO-1 | 优化器状态 | 4× | 与 DP 相同 |
| ZeRO-2 | 优化器状态 + 梯度 | 8× | 与 DP 相同 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 线性扩展 | 1.5× |

**ZeRO-1**：将 Adam 优化器状态（FP32 参数、动量、方差）分片到各 GPU。

**ZeRO-2**：在 ZeRO-1 基础上，也将梯度分片。

**ZeRO-3**：连参数也分片，前向/反向传播时通过 AllGather 收集所需参数。

### 5.3 ZeRO-Offload 与 ZeRO-Infinity

- **ZeRO-Offload**：将优化器状态和梯度卸载到 CPU 内存
- **ZeRO-Infinity**：进一步卸载到 NVMe SSD

### 5.4 DeepSpeed 配置示例

```json
{
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "weight_decay": 0.01
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "contiguous_gradients": true,
        "overlap_comm": true
    },
    "fp16": {
        "enabled": true
    }
}
```

---

## 6. 通信原语

### 6.1 常用集合通信操作

| 操作 | 说明 | 示意图 |
|------|------|--------|
| AllReduce | 所有 GPU 求和后广播 | 用于 DDP 梯度同步 |
| AllGather | 收集所有 GPU 数据并广播 | 用于 ZeRO-3 参数收集 |
| ReduceScatter | 先求和再分散到各 GPU | 用于 ZeRO-2 梯度同步 |
| Broadcast | 从一张 GPU 广播到所有 | 用于参数分发 |
| Scatter / Gather | 分散/收集 | 通用数据分发 |

### 6.2 通信后端

| 后端 | 适用场景 |
|------|---------|
| NCCL | NVIDIA GPU 集群，性能最优 |
| Gloo | CPU 分布式，跨平台 |
| MPI | 传统 HPC 集群 |

---

## 7. 实战配置

### 7.1 单机多卡 (DDP)

```bash
torchrun --nproc_per_node=8 train.py
```

```python
# train.py
import torch.distributed as dist

local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

model = DDP(model, device_ids=[local_rank])
```

### 7.2 多机多卡

```bash
# 主节点
torchrun --nnodes=4 --nproc_per_node=8 --master_addr=10.0.0.1 \
         --master_port=29500 --node_rank=0 train.py

# 其他节点
torchrun --nnodes=4 --nproc_per_node=8 --master_addr=10.0.0.1 \
         --master_port=29500 --node_rank=1 train.py
```

### 7.3 FSDP (Fully Sharded Data Parallel)

PyTorch 原生支持的 ZeRO-3 等价方案：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

### 7.4 大规模训练推荐方案

| 模型规模 | 推荐方案 |
|----------|---------|
| < 1B | DDP |
| 1B - 13B | DeepSpeed ZeRO-2 / FSDP |
| 13B - 70B | DeepSpeed ZeRO-3 + CPU Offload |
| 70B - 175B | Megatron-LM (TP+PP) + DeepSpeed ZeRO |
| > 175B | 3D 并行 + ZeRO-Infinity |

---

## 8. 参考文献

[1] Li, S., et al. PyTorch Distributed: Experiences on Accelerating Data Parallel Training. VLDB, 2020.

[2] Rajbhandari, S., et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC, 2020.

[3] Shoeybi, M., et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053, 2019.

[4] Narayanan, D., et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC, 2021.

[5] Huang, Y., et al. GPipe: Efficient Training of Large Neural Networks using Pipeline Parallelism. NeurIPS, 2019.

[6] Ren, J., et al. ZeRO-Offload: Democratizing Billion-Scale Model Training. ATC, 2021.

[7] Zhao, Y., et al. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. VLDB, 2023.

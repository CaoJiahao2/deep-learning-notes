# 计算机网络基础

> 从 OSI 分层到 TCP/UDP，从 HTTP 到 RDMA —— 理解分布式系统与 AI 集群赖以运行的网络。

---

## 目录

1. [概述](#1)
2. [网络分层模型](#2)
3. [TCP 与 UDP](#3)
4. [IP 与路由](#4)
5. [HTTP 与 API](#5)
6. [AI 场景的网络：RDMA 与集群](#6)
7. [常见网络问题排查](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

计算机网络让设备之间交换数据，是分布式系统、多 GPU 训练、服务部署的基础。对 AI 工程师而言，理解网络有助于：部署推理服务、排查训练通信瓶颈、理解 RDMA 与集群调度（见 `ai-infrastructure.md`）。

本章从分层模型讲起，逐步深入到 AI 最关心的传输层与集群网络。

---

## 2 网络分层模型 {#2}

### 2.1 OSI 七层

| 层 | 名称 | 职责 | 示例 |
|----|------|------|------|
| 7 | 应用层 | 应用协议 | HTTP, DNS, SSH |
| 6 | 表示层 | 数据编码/加密 | TLS 的一部分 |
| 5 | 会话层 | 建立/维护会话 | 很少单独实现 |
| 4 | 传输层 | 端到端传输 | TCP, UDP |
| 3 | 网络层 | 路由与寻址 | IP |
| 2 | 数据链路层 | 相邻节点帧传输 | Ethernet, WiFi |
| 1 | 物理层 | 比特传输 | 网线、光纤 |

### 2.2 TCP/IP 四层（实际使用）

```text
应用层  (HTTP, SSH, NFS)
传输层  (TCP, UDP)
网络层  (IP)
链路层  (Ethernet)
```

### 2.3 分层的好处

每层向上层隐藏细节、向下层提供标准接口，使不同厂商组件可互操作。理解分层有助于定位问题"出在哪一层"。

---

## 3 TCP 与 UDP {#3}

### 3.1 TCP（可靠传输）

- **面向连接**：三次握手建立连接；
- **可靠**：确认、重传、排序；
- **流式**：字节流，无消息边界；
- 适用：需要可靠性的场景（HTTP、SSH、数据库）。

三次握手：

```text
Client → Server: SYN
Server → Client: SYN + ACK
Client → Server: ACK
```

### 3.2 UDP（无连接）

- **无连接**：不建立连接；
- **不可靠**：不保证送达与顺序；
- **低延迟**：无握手开销；
- 适用：实时音视频、DNS、部分分布式通信。

### 3.3 与 AI 的关系

- 多 GPU 训练（NCCL）通常用 RDMA（RoCE/IB），其行为介于 TCP 与 UDP 之间，强调低延迟高带宽；
- 服务部署的 HTTP 通信基于 TCP；
- 了解两者差异有助于诊断"延迟高/丢包"问题。

---

## 4 IP 与路由 {#4}

### 4.1 IP 地址

- **IPv4**：32 位，如 `192.168.1.1`；
- **IPv6**：128 位，如 `2001:db8::1`；
- 私有地址：`10.x`、`172.16-31.x`、`192.168.x`（内网）。

### 4.2 子网与 CIDR

CIDR 用前缀长度表示网络范围：

```text
192.168.1.0/24  → 前 24 位为网络号，256 个地址
```

### 4.3 路由

路由器根据目的 IP 选择下一跳。理解"主机 → 网关 → 路由器 → 目标"的数据路径，有助于排查连通性问题。

### 4.4 常用命令

```bash
ping <host>        # 连通性 + RTT
traceroute <host>  # 路径追踪
ip addr            # 本机 IP
ip route           # 路由表
ss -tulpn          # 监听端口
```

---

## 5 HTTP 与 API {#5}

### 5.1 HTTP 基础

- 请求方法：GET、POST、PUT、DELETE、PATCH；
- 状态码：2xx 成功、4xx 客户端错误、5xx 服务端错误；
- 头部：Content-Type、Authorization、Cache-Control。

### 5.2 RESTful API

以资源为中心的接口设计：

```text
GET    /models/{id}      查询模型
POST   /inference        发起推理请求
DELETE /models/{id}      删除模型
```

### 5.3 与 AI 服务的关系

- 推理服务（见 `model-serving.md`、`llm-ops.md`）通常暴露 HTTP/gRPC 接口；
- 流式输出（SSE / WebSocket）用于逐 token 返回；
- 认证（API Key、OAuth）保护服务。

---

## 6 AI 场景的网络：RDMA 与集群 {#6}

### 6.1 RDMA 简介

RDMA（Remote Direct Memory Access）允许直接访问远端内存，绕过 CPU 与内核协议栈，实现低延迟高带宽。AI 集群（见 `ai-infrastructure.md`）大量使用：

- **InfiniBand**：专用 RDMA 网络，低延迟高带宽；
- **RoCE**：在以太网上实现 RDMA；
- **GPUDirect RDMA**：GPU 显存直接经网卡传输，绕开主机内存。

### 6.2 集合通信

多 GPU 训练依赖集合通信（AllReduce、AllGather、ReduceScatter），由 NCCL 实现（见 `distributed.md`）：

```text
AllReduce: 各节点有部分数据 → 汇总 → 广播给所有节点
```

通信瓶颈直接决定大规模训练效率，网络拓扑（Fat-Tree、Rail-Optimized）至关重要。

### 6.3 分布式系统的网络

- **集群内**：高速 RDMA（训练数据）；
- **集群间**：广域网/对象存储（checkpoint、数据同步）；
- **对外服务**：负载均衡、网关、CDN。

---

## 7 常见网络问题排查 {#7}

| 现象 | 可能原因 | 排查命令 |
|------|---------|---------|
| 无法连通 | 网络/防火墙 | `ping`, `nc -vz host port` |
| 端口拒绝 | 服务未启动 | `ss -tulpn`, `systemctl status` |
| 训练变慢 | 通信瓶颈/丢包 | `nvidia-smi topo -m`, `ibstat`, 带宽测试 |
| 连接超时 | 防火墙/路由 | `traceroute`, `iptables -L` |
| DNS 失败 | DNS 配置 | `nslookup`, `cat /etc/resolv.conf` |

---

## 8 实践建议 {#8}

- 先掌握分层模型与 TCP/UDP 差异，再深入 HTTP 与 RDMA；
- 用 `ping`/`traceroute`/`ss` 等命令排查连通性；
- 推理服务注重 HTTP/gRPC 与流式输出设计；
- 多 GPU 训练关注 NCCL 通信与网络拓扑；
- 分布式系统（Kubernetes）中的 Service、LoadBalancer、NetworkPolicy 都基于网络概念。

---

## 9 参考文献 {#9}

[1] J. Kurose, K. Ross, *Computer Networking: A Top-Down Approach*, Pearson.

[2] A. S. Tanenbaum, D. J. Wetherall, *Computer Networks*, Pearson.

[3] NVIDIA, *GPUDirect RDMA / NCCL Documentation*.

[4] IETF, *RFC 793 (TCP), RFC 768 (UDP), RFC 7230 (HTTP/1.1)*.

[5] Kubernetes 文档: Networking, https://kubernetes.io/docs/concepts/services-networking/

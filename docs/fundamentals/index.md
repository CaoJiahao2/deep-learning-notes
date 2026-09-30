# 基础理论

> 数学、编程与计算机系统 —— 深度学习大厦的地基。

本章节覆盖深度学习所需的全部前置知识：从学习路线与工具环境，到编程与算法基础，再到计算机系统（组成原理、操作系统、网络、数据库、GPU），然后是数学三大支柱（线性代数、概率统计、优化理论），最后落到经典机器学习与其在生产系统中的可靠性。每一篇都以「**在深度学习中如何用到**」为导向组织内容，而不是照搬教材目录。

刚入门时不需要通读全部内容，建议先走 [学习路线图](./learning-roadmap.md) 与 [环境配置](./environment-setup.md)，其余章节按需回查。

## 章节地图

<div class="grid cards" markdown>

-   :material-route:{ .lg .middle } **入门与路线**

    ---

    从零到一的路径规划与工具准备。

    - [学习路线图](./learning-roadmap.md) — 零基础到独立研究的系统化路径
    - [技术演进史 2012-2025](./tech-evolution.md) — 每年热点与标志性论文
    - [环境配置](./environment-setup.md) — GPU 服务器、CUDA、PyTorch

-   :material-code-braces:{ .lg .middle } **编程与算法基础**

    ---

    写得出、跑得快的工程基本功。

    - [Python 与数值计算基础](./python-programming.md) — NumPy 向量化、Pandas
    - [数据结构与算法基础](./data-structures-algorithms.md) — 复杂度、树图、排序搜索

-   :material-cpu-64-bit:{ .lg .middle } **计算机系统**

    ---

    理解程序在硬件上到底发生了什么。

    - [计算机组成原理与层次结构](./computer-architecture.md) — 冯·诺依曼、存储层次
    - [操作系统基础](./operating-system.md) — 进程线程、虚拟内存、并发
    - [计算机网络基础](./computer-networks.md) — TCP/UDP、HTTP、RDMA
    - [数据库基础](./database-basics.md) — SQL、事务、向量数据库
    - [GPU 架构与 CUDA 编程](./gpu-cuda.md) — 线程模型、显存带宽、Kernel 优化

-   :material-function:{ .lg .middle } **数学基础**

    ---

    一切推导与论文阅读的语言。

    - [数学基础](./math-foundations.md) — 线性代数、微积分、信息论
    - [概率论与统计深入](./probability-statistics.md) — 分布族、MLE/MAP、假设检验
    - [优化理论与方法](./optimization-theory.md) — 凸优化、SGD/Adam、二阶方法

-   :material-school:{ .lg .middle } **机器学习**

    ---

    深度学习的母学科与实践起点。

    - [机器学习概述](./machine-learning.md) — 基本概念、三要素、评估方法
    - [ML 系统可靠性](./reliability.md) — 数据漂移、训练-服务偏差

</div>

## 学习建议

1. **先通读全景手册**：[深度学习全景手册](./panorama.md) 把从基础到前沿的全链路串成一条线，先建立地图感，再逐章深入。
2. **数学按需补**：不要试图先把数学全部学完。读 [数学基础](./math-foundations.md) 建立框架，之后在推导遇到障碍时（如反向传播、扩散模型 ELBO）再回到对应小节深挖。
3. **系统知识服务于调优**：GPU 与网络知识在做分布式训练与推理优化时体会最深，可与 [分布式训练](../training/distributed.md)、[推理实现详解](../deployment/inference-implementation.md) 交叉阅读。
4. **动手验证**：每个主题都建议在 [环境配置](./environment-setup.md) 搭好的环境中复现最小示例，看公式落地成代码。

## 延伸阅读

- [深度学习全景手册](./panorama.md) — 全站总纲，从基础贯通到大模型与具身智能
- [模型架构总览](../architectures/index.md) — 基础打好后进入主干网络与生成模型
- [常用工具命令](../resources/tools.md) — Git、Conda、Docker、Linux 速查

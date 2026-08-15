# 数据结构与算法基础

> 从复杂度分析到树与图，从哈希到排序 —— 计算机科学的核心工具，以及它们在深度学习中的实际应用。

---

## 目录

1. [概述](#1)
2. [复杂度分析](#2)
3. [线性数据结构](#3)
4. [哈希表](#4)
5. [树与堆](#5)
6. [图](#6)
7. [排序与搜索](#7)
8. [深度学习中的数据结构](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

数据结构与算法是计算机科学的核心：选择合适的数据结构决定了程序的效率与可扩展性。对 AI 工程师而言，数据结构知识直接服务于：数据加载、张量管理、KV Cache 查找、图神经网络、索引与检索等场景。

核心思想：

```text
程序 = 数据结构 + 算法
```

选择数据结构就是选择"如何在时间与空间之间权衡"。

---

## 2 复杂度分析 {#2}

### 2.1 大 O 记号

衡量算法运行时间随输入规模 $n$ 的增长阶数：

| 记号 | 含义 | 示例 |
|------|------|------|
| $O(1)$ | 常数时间 | 数组按索引访问 |
| $O(\log n)$ | 对数时间 | 二分查找 |
| $O(n)$ | 线性时间 | 遍历数组 |
| $O(n \log n)$ | 线性对数 | 归并/快速排序 |
| $O(n^2)$ | 平方时间 | 双层循环 |
| $O(2^n)$ | 指数时间 | 暴力枚举 |

### 2.2 时间与空间权衡

有时可以用空间换时间，例如：

- 用哈希表把查找从 $O(n)$ 降到 $O(1)$；
- 用缓存（cache）避免重复计算。

### 2.3 平均 vs 最坏

- **最坏复杂度**：保证上界；
- **平均复杂度**：典型情况；
- 分析时应明确讨论的是哪种。

---

## 3 线性数据结构 {#3}

### 3.1 数组（Array）

- 连续内存，按索引访问 $O(1)$；
- 插入/删除中部 $O(n)$；
- 深度学习中的张量底层多为连续数组（见 `python-programming.md`）。

### 3.2 链表（Linked List）

- 节点通过指针相连；
- 插入/删除头部 $O(1)$，按索引访问 $O(n)$；
- 不要求连续内存。

### 3.3 栈（Stack）

- LIFO（后进先出）；
- 应用：函数调用栈、括号匹配、撤销操作、DFS。

### 3.4 队列（Queue）

- FIFO（先进先出）；
- 应用：任务调度、BFS、数据加载缓冲。

### 3.5 Python 中的实现

```python
stack = []              # 用 list 模拟栈
stack.append(1); stack.pop()

from collections import deque
queue = deque()         # 高效队列
queue.append(1); queue.popleft()
```

---

## 4 哈希表 {#4}

### 4.1 原理

哈希表通过哈希函数把键映射到桶（bucket），实现近似 $O(1)$ 的插入、删除、查找：

$$h: \text{key} \to \text{bucket index}$$

### 4.2 冲突处理

- **链地址法**：每个桶存一个链表；
- **开放寻址**：冲突时探测下一个空位。

### 4.3 Python 中的 dict/set

```python
d = {"a": 1, "b": 2}
d.get("c", 0)      # 带默认值
"a" in d           # O(1) 查找

s = {1, 2, 3}      # set 底层也是哈希表
```

### 4.4 应用

- 查找与去重；
- 缓存（LRU 缓存可用哈希表 + 双向链表实现）；
- 稀疏特征/词表映射（token 到 id）。

---

## 5 树与堆 {#5}

### 5.1 二叉树与二叉搜索树

- 二叉搜索树（BST）：左 < 根 < 右，查找 $O(\log n)$（平衡时）；
- 平衡树（AVL、红黑树）保证 $O(\log n)$；
- 应用：符号表、有序集合。

### 5.2 堆（Heap）

- 完全二叉树，父节点优于子节点（最大堆/最小堆）；
- 插入/取最值 $O(\log n)$；
- 应用：优先队列、Top-k 问题、任务调度。

```python
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappop(heap)   # 1
```

### 5.3 树在深度学习中的应用

- 决策树 / 随机森林 / GBDT（见 `machine-learning.md`）；
- AST（抽象语法树）用于代码理解（见 `code-intelligence.md`）；
- 前缀树（Trie）用于 token 与检索（见 `inference-implementation.md` 的 Radix Tree）。

---

## 6 图 {#6}

### 6.1 表示

- **邻接矩阵**：$O(V^2)$ 空间，查找 $O(1)$；
- **邻接表**：$O(V + E)$ 空间，遍历高效。

### 6.2 遍历

- **BFS（广度优先）**：用队列，最短路径；
- **DFS（深度优先）**：用栈/递归，路径探索。

### 6.3 最短路径

- **Dijkstra**：非负权图单源最短路径；
- **Floyd-Warshall**：全源最短路径；
- **Bellman-Ford**：允许负权。

### 6.4 图在深度学习中的应用

- 图神经网络（GNN）的消息传递；
- 知识图谱推理（见 `long-context-rag.md` 的 GraphRAG）；
- 计算图（PyTorch autograd 的 DAG）；
- 分布式训练拓扑。

---

## 7 排序与搜索 {#7}

### 7.1 常见排序

| 算法 | 平均复杂度 | 最坏 | 稳定 | 说明 |
|------|-----------|------|------|------|
| 冒泡 | $O(n^2)$ | $O(n^2)$ | 是 | 教学用 |
| 快排 | $O(n \log n)$ | $O(n^2)$ | 否 | 实际常用 |
| 归并 | $O(n \log n)$ | $O(n \log n)$ | 是 | 稳定 |
| 堆排 | $O(n \log n)$ | $O(n \log n)$ | 否 | 原地 |

### 7.2 查找

- **线性查找**：$O(n)$；
- **二分查找**：有序数组 $O(\log n)$；
- 深度学习常用：在有序列表（如阈值、Top-k）上二分。

---

## 8 深度学习中的数据结构 {#8}

| 场景 | 数据结构 | 作用 |
|------|---------|------|
| 张量存储 | 连续数组 | 底层数据布局（行主序） |
| 词表映射 | 哈希表 | token → id 查找 |
| KV Cache 管理 | 分页块表 | PagedAttention 的逻辑-物理映射 |
| 前缀共享 | 前缀树/Radix Tree | SGLang 的 Prefix Cache |
| 数据加载 | 队列/生产者-消费者 | DataLoader 异步预取 |
| Top-k 采样 | 堆/部分排序 | 从概率分布采样 |
| 图结构数据 | 邻接表/稀疏矩阵 | GNN、知识图谱 |

### 8.1 张量布局

深度学习框架的张量底层是连续一维数组，通过 shape 与 stride 描述维度。理解行主序（row-major）对内存访问优化（见 `gpu-cuda.md`）至关重要。

### 8.2 KV Cache 查找

注意力需要访问历史 KV（见 `inference-implementation.md`），PagedAttention 用块表（类似页表）做逻辑到物理的映射，本质是"哈希 + 分页"思想的结合。

### 8.3 稀疏数据

大规模稀疏特征常用：

- 哈希表存储非零项；
- 稀疏矩阵（CSR/CSC）节省内存；
- 对训练吞吐与显存影响显著（见 `training-pipeline.md`）。

---

## 9 实践建议 {#9}

- 先掌握复杂度分析，再学具体结构；
- 熟用 Python 内置结构：list、dict、set、deque、heapq；
- 面试与工作中常见的"Top-k、LRU、最近邻"问题都依赖特定数据结构；
- 理解底层布局（数组连续性、哈希冲突）对性能优化意义重大；
- 结合深度学习场景学习：KV Cache、DataLoader、图数据能让抽象概念落地。

---

## 10 参考文献 {#10}

[1] T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, MIT Press.

[2] S. S. Skiena, *The Algorithm Design Manual*, Springer.

[3] G. L. Nemhauser, L. A. Wolsey, *Integer and Combinatorial Optimization*, Wiley.

[4] W. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP, 2023.

[5] Python 官方文档: 数据结构, https://docs.python.org/3/tutorial/datastructures.html

# 数据库基础

> 从关系模型与 SQL 到索引与事务，再到向量数据库与 RAG —— 数据管理的核心概念。

---

## 目录

1. [概述](#1)
2. [关系模型与 SQL](#2)
3. [索引原理](#3)
4. [事务与 ACID](#4)
5. [NoSQL 概览](#5)
6. [向量数据库与 RAG](#6)
7. [数据库在 AI 系统中的应用](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

数据库是数据持久化与高效检索的核心组件。对 AI 工程师而言，数据库知识服务于：数据管理、特征存储、日志与元数据、以及检索增强生成（RAG）中的向量检索。

本章从关系模型与 SQL 讲起，覆盖索引、事务、NoSQL，最后落到与 AI 最相关的向量数据库。

---

## 2 关系模型与 SQL {#2}

### 2.1 关系模型

数据组织为表（关系），行（元组）表示记录，列表示字段，主键唯一标识一行。

### 2.2 常用 SQL

```sql
-- 查询
SELECT name, age FROM users WHERE age > 18 ORDER BY age DESC LIMIT 10;

-- 聚合
SELECT dept, COUNT(*) FROM employees GROUP BY dept;

-- 连接
SELECT u.name, o.amount
FROM users u JOIN orders o ON u.id = o.user_id;

-- 插入 / 更新 / 删除
INSERT INTO users (name, age) VALUES ('alice', 30);
UPDATE users SET age = 31 WHERE name = 'alice';
DELETE FROM users WHERE id = 42;
```

### 2.3 规范化

- 第一范式：字段原子化；
- 第二/三范式：消除部分依赖与传递依赖，减少冗余；
- 实际系统常在规范性与查询性能间权衡（反规范化）。

---

## 3 索引原理 {#3}

### 3.1 为什么需要索引

没有索引时查询需全表扫描 $O(n)$；索引让查找接近 $O(\log n)$。

### 3.2 常见索引结构

| 结构 | 特点 | 适用 |
|------|------|------|
| B-Tree / B+Tree | 平衡多路树，磁盘友好 | 范围查询、排序、等值 |
| 哈希索引 | $O(1)$ 等值查找 | 主键/等值查询 |
| 位图索引 | 压缩、高效统计 | 低基数列 |
| 倒排索引 | 词 → 文档列表 | 全文检索 |

B+Tree 是主流关系库默认索引：叶节点有序链表，支持高效范围扫描。

### 3.3 复合索引

对多列建索引时，遵循"最左前缀"原则：

```sql
CREATE INDEX idx_user ON users (dept, age);
-- 可加速 dept、dept+age 查询
```

---

## 4 事务与 ACID {#4}

### 4.1 事务

事务是一组原子操作：要么全部成功，要么全部回滚。

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

### 4.2 ACID

| 特性 | 含义 |
|------|------|
| 原子性 Atomicity | 全部或全不 |
| 一致性 Consistency | 事务前后数据库处于合法状态 |
| 隔离性 Isolation | 并发事务互不干扰 |
| 持久性 Durability | 提交后永久保存 |

### 4.3 隔离级别

- 读未提交 → 读已提交 → 可重复读 → 串行化（隔离性递增，并发性递减）；
- 不同级别对应不同并发异常（脏读、不可重复读、幻读）。

---

## 5 NoSQL 概览 {#5}

| 类型 | 代表 | 特点 | 用途 |
|------|------|------|------|
| 键值 | Redis | 低延迟、内存 | 缓存、会话 |
| 文档 | MongoDB | JSON 灵活 | 半结构化数据 |
| 宽列 | Cassandra | 高吞吐写入 | 海量时序 |
| 图 | Neo4j | 关系查询 | 知识图谱 |

NoSQL 追求水平扩展与灵活 schema，常在一致性上做权衡（CAP 定理）。

---

## 6 向量数据库与 RAG {#6}

### 6.1 为什么需要向量数据库

RAG（见 `long-context-rag.md`）需要在大规模文档集合上做语义近似检索：把文本编码为向量，查询时找最相似的向量。

### 6.2 核心能力

```text
向量索引（HNSW / IVF）
+ 相似度检索（余弦 / 内积 / L2）
+ 元数据过滤
+ 增量更新
```

### 6.3 常见产品

| 系统 | 特点 |
|------|------|
| FAISS | 库级，轻量 |
| Milvus | 分布式向量数据库 |
| pgvector | PostgreSQL 扩展 |
| Qdrant / Weaviate | 全功能向量库 |
| Chroma | 轻量嵌入 |

### 6.4 HNSW 简介

HNSW（Hierarchical Navigable Small World）通过多层级图结构实现近似最近邻搜索，兼顾速度与召回，是向量索引的事实标准之一。

---

## 7 数据库在 AI 系统中的应用 {#7}

| 场景 | 数据 | 存储方案 |
|------|------|---------|
| 训练样本元数据 | 样本 id、标签、路径 | 关系库/对象存储 |
| 特征存储 | 离线/在线特征 | Feature Store |
| 日志与追踪 | 推理日志、指标 | 时序库/对象存储 |
| 应用数据 | 用户、订单 | 关系库 |
| RAG 知识库 | 文档分块 + 向量 | 向量数据库 |

### 7.1 特征存储

训练与推理的特征需保持一致（见 `reliability.md` 的 training-serving skew），Feature Store 统一离线/在线特征，常基于数据库构建。

### 7.2 推理日志

LLM 服务记录请求/响应（见 `llm-ops.md`），用于监控、评测与数据回放，通常落入时序库或对象存储。

---

## 8 实践建议 {#8}

- 掌握核心 SQL 与索引原理，理解查询计划（EXPLAIN）；
- 事务用 ACID 理解一致性，分布式场景用 CAP 权衡；
- RAG 场景关注向量索引（HNSW）与元数据过滤；
- 数据量大时优先考虑分区、索引与缓存；
- 用数据库做元数据管理，用对象存储做大文件（模型、数据集）。

---

## 9 参考文献 {#9}

[1] A. Silberschatz, H. F. Korth, S. Sudarshan, *Database System Concepts*, McGraw-Hill.

[2] R. Ramakrishnan, J. Gehrke, *Database Management Systems*, McGraw-Hill.

[3] J. Johnson, M. Douze, H. Jégou, *Billion-Scale Similarity Search with GPUs (FAISS)*, IEEE Trans. Big Data, 2019.

[4] Y. A. Malkov, D. A. Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs*, IEEE TPAMI, 2018.

[5] PostgreSQL / Redis / Milvus 官方文档.

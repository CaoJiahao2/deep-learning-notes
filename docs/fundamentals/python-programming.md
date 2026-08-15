# Python 与数值计算基础

> 从 Python 语法到 NumPy 向量化，再到 Pandas 数据处理 —— 深度学习开发的第一块编程地基。

---

## 目录

1. [概述](#1)
2. [Python 核心语法](#2)
3. [函数、类与模块](#3)
4. [NumPy：向量化数值计算](#4)
5. [Pandas：数据处理](#5)
6. [代码组织与虚拟环境](#6)
7. [性能与调试](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

Python 是深度学习生态的默认语言：PyTorch、TensorFlow、Transformers 等框架都以 Python 为主要接口。理解 Python 与数值计算基础，是进入深度学习的前提。

本章覆盖：核心语法、函数与类、NumPy 向量化、Pandas 数据处理、虚拟环境与调试。命令行工具（Git/Conda）见 `resources/tools.md`，GPU 相关见 `gpu-cuda.md`。

---

## 2 Python 核心语法 {#2}

### 2.1 变量与数据类型

```python
x = 42          # int
y = 3.14        # float
name = "deep"   # str
flag = True     # bool
lst = [1, 2, 3] # list
dct = {"a": 1}  # dict
tup = (1, 2)    # tuple
st = {1, 2}     # set
```

### 2.2 控制流

```python
for i in range(5):
    if i % 2 == 0:
        print(i)
    else:
        continue

while True:
    break
```

### 2.3 列表推导与生成器

```python
squares = [x ** 2 for x in range(10) if x % 2 == 0]
gen = (x ** 2 for x in range(10))   # 惰性，省内存
```

### 2.4 解包与切片

```python
a, b, *rest = [1, 2, 3, 4]   # a=1, b=2, rest=[3,4]
arr = [0, 1, 2, 3, 4, 5]
arr[1:4]     # [1, 2, 3]
arr[::-1]    # 反转
```

---

## 3 函数、类与模块 {#3}

### 3.1 函数与默认参数

```python
def add(a, b=1):
    """返回 a 与 b 之和。"""
    return a + b

def log(*args, **kwargs):
    print(args, kwargs)
```

### 3.2 lambda 与高阶函数

```python
f = lambda x: x * 2
mapped = list(map(lambda x: x + 1, [1, 2, 3]))
```

### 3.3 类与继承

```python
class Layer:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        raise NotImplementedError

class Linear(Layer):
    def forward(self, x):
        return x  # 示意
```

### 3.4 模块与导入

```python
# file: mymod.py
def helper():
    return 42

# 使用
import mymod
from mymod import helper
from package.sub import item
```

---

## 4 NumPy：向量化数值计算 {#4}

NumPy 提供高效的 $n$ 维数组（ndarray）与向量化运算，是深度学习数值计算的基础。

### 4.1 创建数组

```python
import numpy as np
a = np.array([1, 2, 3])
z = np.zeros((2, 3))
o = np.ones((2, 2))
e = np.eye(3)
r = np.random.randn(3, 4)   # 标准正态
ar = np.arange(10)
```

### 4.2 形状与变形

```python
a = np.arange(12)
a.reshape(3, 4)
a.reshape(-1, 2)   # 自动推断
a.T                # 转置
a.flatten()
```

### 4.3 向量化运算

```python
x = np.random.randn(100, 64)
w = np.random.randn(64, 128)
y = x @ w          # 矩阵乘法
z = np.exp(x)      # 逐元素
s = x.mean(axis=0) # 沿轴归约
```

**为什么向量化重要**：Python 逐元素循环慢；NumPy 底层用 C 实现，且一次操作处理整个数组，避免 Python 解释器开销。深度学习中的矩阵乘法 $Y = XW$ 正是这种向量化计算的核心形态。

### 4.4 广播（Broadcasting）

```python
a = np.zeros((3, 4))
b = np.array([1, 2, 3, 4])   # (4,) 广播到 (3,4)
c = a + b
```

广播规则：从尾部维度比较，维度为 1 或相等时可广播。它避免了显式复制，节省内存与时间。

### 4.5 索引与掩码

```python
a = np.array([[1, 2], [3, 4]])
a[0, 1]           # 2
mask = a > 2
a[mask]           # 布尔索引
```

---

## 5 Pandas：数据处理 {#5}

Pandas 提供 DataFrame 等高层数据结构，用于表格数据处理与预处理。

### 5.1 基本结构

```python
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
s = pd.Series([1, 2, 3])
```

### 5.2 常用操作

```python
df.head()
df.describe()
df["z"] = df["x"] + df["y"]   # 新增列
sub = df[df["x"] > 1]          # 过滤
df.groupby("label").mean()
df.sort_values("x")
```

### 5.3 与 NumPy 互转

```python
arr = df[["x", "y"]].values    # DataFrame -> ndarray
df2 = pd.DataFrame(arr, columns=["x", "y"])
```

### 5.4 缺失值处理

```python
df.isna().sum()
df.dropna()
df.fillna(0)
```

数据预处理（清洗、归一化、划分）是训练流水线的第一步，见 `data-engineering.md` 与 `training-pipeline.md`。

---

## 6 代码组织与虚拟环境 {#6}

### 6.1 项目结构

```text
project/
├── src/            # 源码
│   ├── data.py
│   ├── model.py
│   └── train.py
├── tests/          # 测试
├── configs/        # 配置
├── README.md
└── requirements.txt
```

### 6.2 虚拟环境

虚拟环境隔离依赖，避免版本冲突。常用工具：`venv`、`conda`、`poetry`。命令速查见 `resources/tools.md`。

### 6.3 依赖管理

```text
# requirements.txt
numpy==1.26.0
torch==2.3.0
transformers>=4.40
```

---

## 7 性能与调试 {#7}

### 7.1 性能原则

- 用向量化替代 Python 循环；
- 避免在热循环中重复创建数组；
- 大数据操作优先用 NumPy/Pandas 底层实现；
- 分析用 `cProfile`、`line_profiler`。

### 7.2 调试技巧

```python
import pdb; pdb.set_trace()   # 断点
print(shape, dtype)           # 检查张量
assert x.shape == (64, 128)   # 显式断言
```

### 7.3 常见陷阱

- 可变默认参数：`def f(x=[])` 是共享引用，应改用 `None`；
- NumPy 视图 vs 副本：`a.view()` 与 `a.copy()` 的区别；
- 整数除法：`3 / 2 = 1.5`，`3 // 2 = 1`。

---

## 8 实践建议 {#8}

- 学习路线：语法 → 数据结构 → 函数/类 → NumPy → Pandas → 项目实战；
- 结合路线图"阶段二：编程基础"规划学习进度；
- 用小型深度学习项目（线性回归、MLP）练习 NumPy 向量化；
- 遇到性能问题先 Profile，再优化，避免过早优化；
- 保持代码可复现：固定随机种子、记录依赖版本。

---

## 9 参考文献 {#9}

[1] Guido van Rossum, *The Python Language Reference*, Python Software Foundation.

[2] J. VanderPlas, *Python Data Science Handbook*, O'Reilly, 2016.

[3] W. McKinney, *Python for Data Analysis*, O'Reilly, 2022.

[4] C. R. Harris, K. J. Millman, S. J. van der Walt, et al., *Array Programming with NumPy*, Nature, 2020.

[5] NumPy / Pandas 官方文档, https://numpy.org/, https://pandas.pydata.org/

# 反向传播：从链式法则到完整推导

> 神经网络训练的数学核心。本文将推导多层感知机（MLP）的反向传播，并给出 softmax + 交叉熵、MSE 的完整梯度。

---

## 目录

1. [记号约定](#1)
2. [前向传播](#2)
3. [链式法则回顾](#3)
4. [单层梯度推导](#4)
5. [softmax + 交叉熵梯度](#5-softmax)
6. [向量化反向传播](#6)
7. [梯度消失与爆炸](#7)
8. [总结](#8)

---

## 1. 记号约定

考虑一个 $L$ 层的 MLP。第 $\ell$ 层：

- 输入 $\mathbf{a}^{\ell-1} \in \mathbb{R}^{d_{\ell-1}}$
- 权重 $\mathbf{W}^{\ell} \in \mathbb{R}^{d_{\ell} \times d_{\ell-1}}$，偏置 $\mathbf{b}^{\ell} \in \mathbb{R}^{d_\ell}$
- 线性组合 $\mathbf{z}^{\ell} = \mathbf{W}^{\ell}\mathbf{a}^{\ell-1} + \mathbf{b}^{\ell}$
- 激活输出 $\mathbf{a}^{\ell} = \sigma(\mathbf{z}^{\ell})$（$\sigma$ 为逐元素激活函数，如 ReLU / sigmoid / tanh）

记损失为标量 $\mathcal{L}$。输入层 $\mathbf{a}^0 = \mathbf{x}$，输出层 $\mathbf{a}^L = \hat{\mathbf{y}}$。

**关键定义**：误差项（error term）

$$\boldsymbol{\delta}^{\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{\ell}}$$

- $\boldsymbol{\delta}^{L}$ 是输出层误差项，由损失函数直接给出；
- 更前层的 $\boldsymbol{\delta}^{\ell-1}$ 通过 $\boldsymbol{\delta}^{\ell}$ 递推得到。

---

## 2. 前向传播

$$\mathbf{z}^{\ell} = \mathbf{W}^{\ell}\mathbf{a}^{\ell-1} + \mathbf{b}^{\ell}, \qquad \mathbf{a}^{\ell} = \sigma(\mathbf{z}^{\ell})$$

最终损失 $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$。例如 softmax + 交叉熵：

$$\hat{y}_c = \frac{e^{z^L_c}}{\sum_k e^{z^L_k}}, \qquad \mathcal{L} = -\sum_c y_c \log \hat{y}_c$$

---

## 3. 链式法则回顾

反向传播的核心：**从输出层开始，逐层将梯度"传回去"**。

对任意层 $\ell$，权重梯度为：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{\ell}} \cdot \frac{\partial \mathbf{z}^{\ell}}{\partial \mathbf{W}^{\ell}} = \boldsymbol{\delta}^{\ell} \left(\mathbf{a}^{\ell-1}\right)^\top$$

```text
损失 L  ←  δ^L ← γ^L
   ↑                  ↑
z^L = W^L a^{L-1} + b^L
   ↑
a^{L-1} = σ(z^{L-1})
   ...逐层向前
```

---

## 4. 单层梯度推导

### 4.1 输出层（最后一层）

对任意输出单元 $c$，若损失为交叉熵且输出为 softmax，可证明（见 §5）：

$$\delta^{L}_c = \frac{\partial \mathcal{L}}{\partial z^{L}_c} = \hat{y}_c - y_c$$

即输出层误差 = 预测概率 − 真实标签（one-hot）。

权重梯度：

$$\frac{\partial \mathcal{L}}{\partial W^{L}_{cj}} = \delta^{L}_c \, a^{L-1}_j, \qquad \frac{\partial \mathcal{L}}{\partial b^{L}_c} = \delta^{L}_c$$

### 4.2 隐藏层——误差回传

由 $\mathbf{z}^{\ell} = \mathbf{W}^{\ell}\sigma(\mathbf{z}^{\ell-1}) + \mathbf{b}^{\ell}$，用链式法则：

$$\delta^{\ell-1}_j = \frac{\partial \mathcal{L}}{\partial z^{\ell-1}_j} = \sum_i \frac{\partial \mathcal{L}}{\partial z^{\ell}_i} \cdot \frac{\partial z^{\ell}_i}{\partial a^{\ell-1}_j} \cdot \frac{\partial a^{\ell-1}_j}{\partial z^{\ell-1}_j}$$

其中 $\dfrac{\partial z^{\ell}_i}{\partial a^{\ell-1}_j} = W^{\ell}_{ij}$，$\dfrac{\partial a^{\ell-1}_j}{\partial z^{\ell-1}_j} = \sigma'(z^{\ell-1}_j)$。因此：

$$\delta^{\ell-1}_j = \sigma'(z^{\ell-1}_j) \sum_{i} W^{\ell}_{ij} \delta^{\ell}_i$$

**向量形式**：

$$\boldsymbol{\delta}^{\ell-1} = \left(\mathbf{W}^{\ell}\right)^\top \boldsymbol{\delta}^{\ell} \odot \sigma'(\mathbf{z}^{\ell-1})$$

其中 $\odot$ 为逐元素（Hadamard）乘积。

### 4.3 权重与偏置梯度

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell-1}} = \boldsymbol{\delta}^{\ell-1} \left(\mathbf{a}^{\ell-2}\right)^\top, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{\ell-1}} = \boldsymbol{\delta}^{\ell-1}$$

---

## 5. softmax + 交叉熵梯度

### 5.1 softmax 的雅可比

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

对 $z_j$ 求偏导（分 $i = j$ 与 $i \neq j$ 两种情况）：

- 当 $i = j$：$\dfrac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$
- 当 $i \neq j$：$\dfrac{\partial p_i}{\partial z_j} = -p_i p_j$

合并为：

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

其中 $\delta_{ij}$ 为 Kronecker delta。

### 5.2 交叉熵损失对 logits 的梯度

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial p_i} \frac{\partial p_i}{\partial z_j} = -\sum_i \frac{y_i}{p_i} \cdot p_i(\delta_{ij} - p_j) = -\sum_i y_i(\delta_{ij} - p_j)$$

由于 $\sum_i y_i = 1$（one-hot 或概率标签），化简得：

$$\frac{\partial \mathcal{L}}{\partial z_j} = -y_j + p_j \sum_i y_i = p_j - y_j$$

这正是 §4.1 的结论：**输出误差 = 预测概率 − 标签**。

---

## 6. 向量化反向传播

对 mini-batch $\mathbf{X} \in \mathbb{R}^{B \times d_0}$，逐层计算：

```text
# 前向
A[0] = X
for l in 1..L:
    Z[l] = A[l-1] @ W[l].T + b[l]
    A[l] = act(Z[l])

# 反向
δ[L] = A[L] - Y                       # softmax+CE 输出层
for l in L-1 .. 1:
    δ[l] = (δ[l+1] @ W[l+1]) * act'(Z[l])
    dW[l+1] = δ[l+1].T @ A[l] / B
    db[l+1] = δ[l+1].mean(axis=0)
```

**内存/计算复杂度**：每层需保存激活值 $\mathbf{a}^{\ell}$ 用于反向传播，这就是训练显存远大于推理的原因；激活值检查点（activation checkpointing）通过"反向时重算前向"来降低显存。

---

## 7. 梯度消失与爆炸

将误差回传公式展开 $k$ 层：

$$\boldsymbol{\delta}^{L-k} = \left(\prod_{\ell=L-k+1}^{L} \left(\mathbf{W}^{\ell}\right)^\top \operatorname{diag}\left(\sigma'(\mathbf{z}^{\ell-1})\right)\right) \boldsymbol{\delta}^{L}$$

- 若 $\sigma$ 为 sigmoid/tanh，其导数上界 $< 1$（sigmoid 最大 $1/4$），深层时误差指数衰减 → **梯度消失**；
- 若 $\mathbf{W}$ 的最大特征值 $> 1$，误差指数放大 → **梯度爆炸**。

**缓解手段**：
- 激活函数：ReLU 导数恒为 1（正区间），缓解消失；
- 门控：LSTM 的 CEC（见 `architectures/rnn-lstm.md`）；
- 残差连接：恒等路径使梯度直通（见 `architectures/transformer.md`）；
- 归一化：BN / LN / RMSNorm 稳定激活分布；
- 优化器：梯度裁剪、自适应学习率。

---

## 8. 总结

| 量 | 公式 |
|----|------|
| 输出层误差 | $\boldsymbol{\delta}^{L} = \hat{\mathbf{y}} - \mathbf{y}$（softmax + CE） |
| 误差回传 | $\boldsymbol{\delta}^{\ell-1} = (\mathbf{W}^{\ell})^\top \boldsymbol{\delta}^{\ell} \odot \sigma'(\mathbf{z}^{\ell-1})$ |
| 权重梯度 | $\dfrac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell}} = \boldsymbol{\delta}^{\ell} (\mathbf{a}^{\ell-1})^\top$ |
| 偏置梯度 | $\dfrac{\partial \mathcal{L}}{\partial \mathbf{b}^{\ell}} = \boldsymbol{\delta}^{\ell}$ |

反向传播不是一个"新算法"，而是对由可微操作组成的计算图反复应用链式法则。现代深度学习框架（PyTorch / TensorFlow）通过**自动微分**（autograd，反向模式）自动完成这一切，但理解其推导是调试、设计新模块与理解梯度现象的前提。

---

## 参考文献

[1] Rumelhart, D., Hinton, G. & Williams, R. Learning representations by back-propagating errors. Nature, 1986.

[2] Goodfellow, I., Bengio, Y. & Courville, A. Deep Learning. MIT Press, 2016. (Chapter 6: Deep Feedforward Networks)

[3] Baydin, A., et al. Automatic Differentiation in Machine Learning: A Survey. JMLR, 2018.

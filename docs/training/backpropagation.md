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
8. [PyTorch autograd 深入](#8)
9. [工程实践：显存、梯度累积与常见陷阱](#9)
10. [总结](#10)

---

## 1. 记号约定 {#1}

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

## 2. 前向传播 {#2}

$$\mathbf{z}^{\ell} = \mathbf{W}^{\ell}\mathbf{a}^{\ell-1} + \mathbf{b}^{\ell}, \qquad \mathbf{a}^{\ell} = \sigma(\mathbf{z}^{\ell})$$

最终损失 $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$。例如 softmax + 交叉熵：

$$\hat{y}_c = \frac{e^{z^L_c}}{\sum_k e^{z^L_k}}, \qquad \mathcal{L} = -\sum_c y_c \log \hat{y}_c$$

---

## 3. 链式法则回顾 {#3}

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

### 3.1 计算图：把链式法则变成算法

反向传播的可执行形式建立在**计算图**（computational graph）上：把前向计算拆成原子操作，每个中间变量是一个节点，"参与了哪个运算"构成有向边。以单层 + 平方损失为例：

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \qquad \mathbf{a} = \sigma(\mathbf{z}), \qquad \mathcal{L} = \frac{1}{2}\|\mathbf{a} - \mathbf{y}\|^2$$

反向传播 = 沿边逆行，把"损失对下游的梯度"经过每个节点的**局部导数（VJP）**换算到输入并**累加**。为此每个节点需保存两样东西：前向值（如 $\sigma'$ 必须在 $\mathbf{z}$ 处取值）与 VJP 闭包。当一个变量被多个下游使用（权重共享、复用的中间结果）时，各路径梯度相加——多元链式法则的全导数公式，这也是每个节点必须等所有消费者的梯度到齐才能求值的原因。

### 3.2 拓扑排序与反向模式自动微分

- **前向传播按拓扑序（topological order）求值**（计算任一节点时其全部父节点已就绪），**反向传播按逆拓扑序遍历**（求任一节点梯度时其全部子节点梯度已到齐）。计算图是 DAG，拓扑序必然存在（Kahn 算法 / DFS 均可），求值复杂度 $O(V+E)$；
- **反向模式（reverse mode）AD**：训练时输出是标量损失、参数动辄上亿维，一次反向遍历即可同时得到所有 $\partial \mathcal{L}/\partial \theta_i$，代价约为一次前向；数值微分需 $n$ 次前向、前向模式 AD 需 $n$ 次传递——这就是训练一律采用反向模式的根本原因。

---

## 4. 单层梯度推导 {#4}

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

### 4.4 维度自查：实现排错的第一手段

实现时逐条核对形状（梯度必须与参数同形；维度对不上几乎总意味着公式记错，例如把外积写成了内积）：

| 公式 | 形状变化 | 结果形状 |
|------|----------|----------|
| $\boldsymbol{\delta}^{\ell-1} = (\mathbf{W}^{\ell})^\top \boldsymbol{\delta}^{\ell} \odot \sigma'(\mathbf{z}^{\ell-1})$ | $(d_{\ell-1} \times d_\ell)(d_\ell \times 1)$ | $d_{\ell-1} \times 1$ |
| $\dfrac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell}} = \boldsymbol{\delta}^{\ell} (\mathbf{a}^{\ell-1})^\top$ | $(d_\ell \times 1)(1 \times d_{\ell-1})$ | $d_\ell \times d_{\ell-1}$ |

---

## 5. softmax + 交叉熵梯度 {#5-softmax}

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

### 5.3 数值稳定性：log-sum-exp 与融合实现

按定义计算 softmax 会在 $z_c$ 较大时溢出（$e^{1000}$ 即 inf）。利用平移不变性减去最大 logit，结果不变而指数参数被压到 $(-\infty, 0]$：

$$\hat{y}_c = \frac{e^{z_c - m}}{\sum_k e^{z_k - m}}, \qquad m = \max_k z_k$$

交叉熵同理不要先算概率再取对数（概率下溢为 0 后 $\log$ 得 $-\infty$），应使用 log-sum-exp 的稳定形式：

$$\log \hat{y}_c = z_c - m - \log \sum_k e^{z_k - m}$$

实践要点：`F.cross_entropy` 等 fused 实现把 softmax、log 与交叉熵合并，反向直接用 $\hat{y}_c - y_c$（§5.2），既避免中间量下溢又省去构造雅可比——联合推导不仅是数学简洁，更是工程必需。

---

## 6. 向量化反向传播 {#6}

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

**内存/计算复杂度**：每层需保存激活值 $\mathbf{a}^{\ell}$ 用于反向传播，这就是训练显存远大于推理的原因；激活值检查点（activation checkpointing）通过"反向时重算前向"来降低显存（详见 §9.1）。

### 6.1 mini-batch 矩阵形式的完整推导

把单个样本堆成 batch：$\mathbf{A}^{\ell-1} \in \mathbb{R}^{B \times d_{\ell-1}}$（每行一个样本），前向为

$$\mathbf{Z}^{\ell} = \mathbf{A}^{\ell-1}\left(\mathbf{W}^{\ell}\right)^\top + \mathbf{b}^{\ell}, \qquad \mathbf{A}^{\ell} = \sigma\left(\mathbf{Z}^{\ell}\right)$$

（$\mathbf{b}^{\ell}$ 按行广播）。记 batch 误差矩阵 $\boldsymbol{\Delta}^{\ell} = \partial \mathcal{L}/\partial \mathbf{Z}^{\ell} \in \mathbb{R}^{B \times d_\ell}$，逐元素写开再堆叠，得到与 §4 完全同构的矩阵方程：

$$\boldsymbol{\Delta}^{\ell-1} = \left(\boldsymbol{\Delta}^{\ell}\mathbf{W}^{\ell}\right) \odot \sigma'\left(\mathbf{Z}^{\ell-1}\right)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell}} = \frac{1}{B}\left(\boldsymbol{\Delta}^{\ell}\right)^\top \mathbf{A}^{\ell-1}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{\ell}} = \frac{1}{B}\left(\boldsymbol{\Delta}^{\ell}\right)^\top \mathbf{1}_B$$

其中 $\mathbf{1}_B$ 为 $B$ 维全 1 向量。核对：$\left(\boldsymbol{\Delta}^{\ell}\right)^\top \mathbf{A}^{\ell-1}$ 的 $(i,j)$ 元是 $\sum_b \Delta^{\ell}_{b,i} A^{\ell-1}_{b,j}$，恰为逐样本梯度 $\delta^{\ell}_i a^{\ell-1}_j$ 的 batch 求和——**批量化只是把逐样本公式按行堆叠**。

### 6.2 NumPy 参考实现

```python
import numpy as np

def forward(X, params):
    acts, masks, A = [X], [], X        # 保存每层激活与 ReLU 掩码（反向要用）
    for W, b in params[:-1]:           # 隐藏层
        A = np.maximum(A @ W.T + b, 0.0)   # ReLU（掩码等价于 A > 0）
        acts.append(A); masks.append(A > 0)
    acts.append(A @ params[-1][0].T + params[-1][1])  # 输出层 logits
    return acts, masks

def backward(acts, masks, Y, params):
    B = Y.shape[0]
    ZL = acts[-1] - acts[-1].max(axis=1, keepdims=True)  # logit 平移，见 5.3
    P = np.exp(ZL); P /= P.sum(axis=1, keepdims=True)    # softmax
    delta = (P - Y) / B                                  # 输出层误差
    grads = []
    for l in range(len(params) - 1, -1, -1):
        grads.append((delta.T @ acts[l], delta.sum(axis=0)))  # dW, db
        if l > 0:
            delta = (delta @ params[l][0]) * masks[l - 1]     # 误差回传
    return grads[::-1], P
```

### 6.3 梯度检查（gradient checking）

autograd 时代依然必备的单元测试手段：用**中心差分**独立估计梯度，与解析梯度对比。

$$g^{\text{num}}_i = \frac{\mathcal{L}(\theta_i + \varepsilon) - \mathcal{L}(\theta_i - \varepsilon)}{2\varepsilon}, \qquad \varepsilon \approx 10^{-5}$$

中心差分截断误差 $O(\varepsilon^2)$（单侧差分仅 $O(\varepsilon)$），$\varepsilon$ 过小则被 float32 舍入误差支配。比较用相对误差 $\|\mathbf{g}^{\text{ana}} - \mathbf{g}^{\text{num}}\|_2 / (\|\mathbf{g}^{\text{ana}}\|_2 + \|\mathbf{g}^{\text{num}}\|_2)$，经验判据：$< 10^{-7}$ 通过；$10^{-5}$ 量级需排查（ReLU 不可导点、dropout 随机性、float16）；$> 10^{-3}$ 几乎必有 bug。注意用双精度、固定随机性（eval 模式、固定 seed）、避开不可导点；$O(n)$ 的成本只用于小规模单元测试。

```python
def numerical_grad(f, theta, eps=1e-5):
    # 中心差分数值梯度：f(theta) 返回标量损失
    g = np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i], tm[i] = tp[i] + eps, tm[i] - eps
        g[i] = (f(tp) - f(tm)) / (2 * eps)
    return g
```

---

## 7. 梯度消失与爆炸 {#7}

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

### 7.1 RNN 的 BPTT：消失/爆炸的原型场景

BPTT（backpropagation through time）把 RNN 沿时间展开成"深度 = 序列长度"的网络，且各时间步**共享**同一组权重：

$$\mathbf{h}_t = \tanh\left(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}\right)$$

误差逐时间步回传的递推式与前馈网络逐层回传同构：$\boldsymbol{\delta}_t = \mathbf{W}_{hh}^\top \boldsymbol{\delta}_{t+1} \odot \tanh'(\mathbf{z}_t)$。对 $\mathbf{W}_{hh}$ 的梯度需对所有时间间隔 $(t, k)$ 求和，跨距为 $t-k$ 的路径上出现连乘：

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \operatorname{diag}\left(\tanh'(\mathbf{z}_i)\right)\mathbf{W}_{hh}$$

由于 $\tanh' \le 1$：$\mathbf{W}_{hh}$ 最大奇异值 $\rho < 1$ 时梯度随 $t-k$ 指数衰减（消失），$\rho > 1$ 时指数放大（爆炸），序列越长越尖锐——这正是 RNN 难以学到长距离依赖的原因。对策与前馈情形一致但侧重不同：**爆炸好处理**，梯度裁剪即可稳定训练；**消失是架构问题**，需 LSTM/GRU 的门控加法路径让误差近似恒等直通，或改用注意力；也可用**截断 BPTT（TBPTT）**只回传固定窗口，牺牲长程信用分配换取可行性与显存。

---

## 8. PyTorch autograd 深入 {#8}

### 8.1 动态图与反向模式自动微分

PyTorch 采用 **define-by-run** 动态图：前向执行到哪里，计算图就即时构建到哪里——每个 tensor 的 `.grad_fn` 记录"由什么操作、哪些输入生成"，连成 DAG；控制流（`if` / `for`、变长序列）每次运行都按真实执行路径建图，无需预先声明静态结构（对比 TensorFlow 1.x 的 graph mode）。反向传播即在这张 DAG 上按**逆拓扑序**（§3.2）遍历，每个节点的 backward 是一个向量-雅可比积（VJP）：

$$\bar{\mathbf{x}} = \mathbf{J}^\top \bar{\mathbf{y}}$$

autograd 从不显式构造雅可比矩阵——每个内置 op 注册自己的 VJP 闭包（如逐元素乘法的 VJP 就是与另一因子做 $\odot$），内存中只保存前向值与闭包。

### 8.2 `torch.autograd.grad` 与 `.backward()` 的分工

```python
x = torch.randn(4, 3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
y = (x @ W.T).square().sum()

gx, gW = torch.autograd.grad(y, (x, W), retain_graph=True)  # 梯度作为返回值
y.backward()                                                # 梯度累加进叶子 .grad
```

- `.backward()`：把梯度**累加**进各**叶子**张量的 `.grad`（故每步前要 `optimizer.zero_grad()`），对非标量输出需传同形 `gradient` 权重；适合训练循环；
- `torch.autograd.grad(outputs, inputs)`：即时返回梯度、不写 `.grad`，适合只对个别参数求导、对中间量求导、Hessian-vector 积等高阶运算。两者底层走同一条反向遍历，只是"结果去向"不同。

### 8.3 backward 钩子：窥视与干预反向传播

```python
hook = W.register_hook(lambda g: print("grad norm:", float(g.norm())))
(x @ W).square().mean().backward()
hook.remove()                          # 用完及时移除，避免泄漏
```

钩子最适合监控每层梯度范数——定位梯度消失/爆炸最直接的手段；其返回值会替换该梯度，可用于自定义裁剪、梯度掩码（gradient masking）。模块级可用 `register_full_backward_hook` 同时观察一层的输入与输出梯度。

### 8.4 梯度累积：小显存模拟大 batch

`.grad` 是累加语义，天然支持把大 batch 拆成 $k$ 份 micro-batch 依次回传，等价于对完整 batch 求梯度 $\nabla_{\theta}\,\mathcal{L} = \frac{1}{k}\sum_{i=1}^{k}\nabla_{\theta}\,\mathcal{L}_i$：

```python
accum = 4                                   # 4 个 micro-batch 模拟大 batch
for i, (xb, yb) in enumerate(loader):
    loss = loss_fn(model(xb), yb) / accum   # 预先除以累积步数
    loss.backward()                         # .grad 逐次累加
    if (i + 1) % accum == 0:
        optimizer.step(); optimizer.zero_grad()
```

注意 BatchNorm 统计量只反映每个 micro-batch（可用 SyncBN 等方案缓解）；梯度累积与激活检查点（§9.1）搭配是大模型微调的标配组合。

---

## 9. 工程实践：显存、梯度累积与常见陷阱 {#9}

### 9.1 激活检查点（activation checkpointing）

训练显存 ≈ 参数 + 优化器状态 + **激活缓存**；反向需要每层激活（§6 的 cache），Transformer 的激活随层数与序列长度线性增长，往往才是显存大头。checkpointing 的思路：前向时**只保存分段的边界激活**，段内中间量丢弃；反向到该段时用边界激活**重算**段内前向再回传。每段约 $\sqrt{L}$ 层、共 $\sqrt{L}$ 个边界时，激活显存从 $O(L)$ 降到 $O(\sqrt{L})$，代价约为一次额外前向（step 慢 25%–40%）。

| 策略 | 激活显存 | 额外计算 |
|------|----------|----------|
| 全缓存（默认） | $O(L)$ | 无 |
| 全部分段检查点 | $O(\sqrt{L})$ | 约一次前向 |
| 只检查点部分层 | 介于两者 | 按比例减少 |

PyTorch 中的用法是把段内前向包进 `torch.utils.checkpoint.checkpoint(inner, x, use_reentrant=False)`。细节：默认 `preserve_rng_state=True` 会重放 RNG 状态，保证 dropout 重算与首次前向一致；`use_reentrant=False` 是当前推荐配置，对嵌套段兼容性更好。

### 9.2 常见陷阱清单

- **in-place 操作破坏计算图**：反向所需的中间值被覆盖时报错 "modified by an inplace operation"。autograd 用版本计数器（`tensor._version`）检测——每次 in-place 修改版本号加一，backward 时发现所需张量版本不符即报错。高风险写法：`x += y`、`relu_()`、掩码赋值，以及 `inplace=True` 的激活作用在反向仍需要的张量上。
- **detach 的时机**：忘 detach——把带图的 loss 存入日志/指标统计，计算图与全部中间激活被引用，显存随步数上涨（应 `loss.item()` 或 `.detach()`）；反向又该 detach——RL 的 target 网络、EMA teacher、对比学习负样本队列等不该收梯度的分支忘 detach，梯度会流入错误分支或直接报错。
- **非叶子节点没有 `.grad`**：backward 后中间结果为 `None`，调试用 `z.retain_grad()` 或改用 `torch.autograd.grad(loss, z)`。验证与推理包在 `with torch.no_grad():` 中；`.eval()` 关闭 dropout 与 BN 更新（也是梯度检查的前提）。

---

## 10. 总结 {#10}

| 量 | 公式 |
|----|------|
| 输出层误差 | $\boldsymbol{\delta}^{L} = \hat{\mathbf{y}} - \mathbf{y}$（softmax + CE） |
| 误差回传 | $\boldsymbol{\delta}^{\ell-1} = (\mathbf{W}^{\ell})^\top \boldsymbol{\delta}^{\ell} \odot \sigma'(\mathbf{z}^{\ell-1})$ |
| 权重梯度 | $\dfrac{\partial \mathcal{L}}{\partial \mathbf{W}^{\ell}} = \boldsymbol{\delta}^{\ell} (\mathbf{a}^{\ell-1})^\top$ |
| 偏置梯度 | $\dfrac{\partial \mathcal{L}}{\partial \mathbf{b}^{\ell}} = \boldsymbol{\delta}^{\ell}$ |

反向传播不是一个"新算法"，而是对由可微操作组成的计算图反复应用链式法则。现代深度学习框架（PyTorch / TensorFlow）通过**自动微分**（autograd，反向模式）自动完成这一切，但理解其推导是调试、设计新模块与理解梯度现象的前提。从数学（§3–§7：计算图、逐层推导、BPTT）到实现（§8–§9：autograd、显存与陷阱），主线始终是同一句话：**在计算图上按逆拓扑序做向量-雅可比积**。

---

## 参考文献

[1] Rumelhart, D., Hinton, G. & Williams, R. Learning representations by back-propagating errors. Nature, 1986.

[2] Goodfellow, I., Bengio, Y. & Courville, A. Deep Learning. MIT Press, 2016. (Chapter 6: Deep Feedforward Networks)

[3] Baydin, A., et al. Automatic Differentiation in Machine Learning: A Survey. JMLR, 2018.

[4] Pascanu, R., Mikolov, T. & Bengio, Y. On the Difficulty of Training Recurrent Neural Networks. ICML, 2013.

[5] Paszke, A., et al. Automatic Differentiation in PyTorch. NeurIPS Autodiff Workshop, 2017.

[6] Chen, T., et al. Training Deep Nets with Sublinear Memory Cost. arXiv, 2016.

# 数学基础

> 深度学习推导所需的线性代数、微积分、概率与信息论基础。后续所有文档的公式都建立在本章符号与恒等式之上。

---

## 目录

1. [线性代数](#1)
2. [微积分与优化](#2)
3. [概率与统计](#3)
4. [信息论](#4)
5. [常用恒等式](#5)
6. [参考文献](#6)

---

## 1. 线性代数

### 1.1 向量与矩阵内积

两个向量 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ 的内积：

$$\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta$$

其中 $\theta$ 是两个向量之间的夹角。内积越大，两向量越"相似"（方向越一致）。

### 1.2 矩阵乘法与向量化

对 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 与 $\mathbf{W} \in \mathbb{R}^{d \times h}$，$\mathbf{Y} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{n \times h}$：

$$Y_{ij} = \sum_{k=1}^{d} X_{ik} W_{kj}$$

**矩阵迹（Trace）的循环性质**（在梯度推导中反复用到）：

$$\operatorname{Tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{Tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \operatorname{Tr}(\mathbf{B}\mathbf{C}\mathbf{A})$$

### 1.3 特征值与奇异值

- **特征值分解**：对称矩阵 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$，$\mathbf{Q}$ 为正交矩阵，$\boldsymbol{\Lambda}$ 为对角特征值矩阵。
- **奇异值分解（SVD）**：任意矩阵 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，$\boldsymbol{\Sigma}$ 的对角元为奇异值。SVD 是 PCA、低秩近似（LoRA 的理论基础）的核心工具。

**低秩近似（Eckart–Young 定理）**：对秩为 $r$ 的矩阵，其最佳 $k$ 秩近似（Frobenius 范数意义下）为截断 SVD：

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top, \qquad \min_{\operatorname{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

这是 LoRA 假设"权重更新 $\Delta W$ 是低秩的"的数学依据。

### 1.4 矩阵求导

深度学习反向传播的核心是**标量对矩阵的梯度**。设 $L$ 为标量损失：

- $L = \mathbf{a}^\top \mathbf{x}$，则 $\dfrac{\partial L}{\partial \mathbf{x}} = \mathbf{a}$
- $L = \mathbf{x}^\top \mathbf{A} \mathbf{x}$（$\mathbf{A}$ 对称），则 $\dfrac{\partial L}{\partial \mathbf{x}} = 2\mathbf{A}\mathbf{x}$
- $L = \|\mathbf{x}\|_2^2 = \mathbf{x}^\top \mathbf{x}$，则 $\dfrac{\partial L}{\partial \mathbf{x}} = 2\mathbf{x}$

---

## 2. 微积分与优化

### 2.1 链式法则

设 $z = f(y)$，$y = g(x)$，则：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

对向量：$z = f(\mathbf{y})$，$\mathbf{y} = g(\mathbf{x})$，则 Jacobian 形式：

$$\frac{\partial z}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^\top \frac{\partial z}{\partial \mathbf{y}}$$

反向传播正是从输出层到输入层反复应用链式法则。

### 2.2 泰勒展开

函数 $f$ 在 $x_0$ 附近的二阶泰勒展开：

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2}f''(x_0)(x - x_0)^2$$

对多元函数 $f(\mathbf{x})$（$\mathbf{H}$ 为 Hessian 矩阵）：

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^\top \mathbf{H}(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$$

### 2.3 凸性与梯度下降收敛

对 $L$-光滑（梯度 Lipschitz）且 $\mu$-强凸的函数，梯度下降

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

取最优步长 $\eta = \dfrac{2}{L + \mu}$ 时，收敛率为线性收敛：

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t} \left(f(\mathbf{x}_0) - f(\mathbf{x}^*)\right)$$

其中条件数 $\kappa = L / \mu$：$\kappa$ 越小（损失曲面越"圆"）收敛越快。

### 2.4 拉格朗日与 KKT

求解带约束优化 $\min f(\mathbf{x}) \ \text{s.t.} \ g(\mathbf{x}) = 0$，构造拉格朗日函数：

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

一阶条件 $\nabla_{\mathbf{x}}\mathcal{L} = 0, \nabla_\lambda \mathcal{L} = 0$ 给出驻点。这是 SVM 对偶、WGAN 对偶推导的基础。

---

## 3. 概率与统计

### 3.1 条件概率与贝叶斯定理

$$P(A \mid B) = \frac{P(A, B)}{P(B)}, \qquad P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

### 3.2 期望与方差

$$\mathbb{E}[X] = \sum_x x P(x), \qquad \operatorname{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2$$

**方差分解（Law of Total Variance）**：

$$\operatorname{Var}[X] = \mathbb{E}[\operatorname{Var}[X \mid Y]] + \operatorname{Var}[\mathbb{E}[X \mid Y]]$$

### 3.3 最大似然估计（MLE）与交叉熵

给定数据 $\{x_i\}_{i=1}^N$ 与参数化分布 $p_\theta$，对数似然：

$$\ell(\theta) = \sum_{i=1}^{N} \log p_\theta(x_i)$$

MLE：$\hat{\theta} = \arg\max_\theta \ell(\theta)$。

**分类中的交叉熵**：对 one-hot 标签 $\mathbf{y}$ 与模型输出概率 $\hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{z})$，

$$\mathcal{L}_{\text{CE}} = -\sum_{c} y_c \log \hat{y}_c = -\log \hat{y}_{k}$$

其中 $k$ 为真实类别下标。这正是对真实类别取负对数概率，等价于最大化该类的对数似然。

### 3.4 高斯分布

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

多元高斯：

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**高斯重参数化**：若 $z \sim \mathcal{N}(\mu, \sigma^2)$，可写成 $z = \mu + \sigma \epsilon$，$\epsilon \sim \mathcal{N}(0, 1)$。这是 VAE 与扩散模型"噪声添加"的数学基础。

---

## 4. 信息论

### 4.1 熵、联合熵与条件熵

$$H(X) = -\sum_x p(x) \log p(x), \qquad H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

$$H(Y \mid X) = -\sum_{x,y} p(x,y) \log p(y \mid x)$$

### 4.2 KL 散度

$$D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right] = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

性质：$D_{KL}(P \parallel Q) \ge 0$，当且仅当 $P = Q$ 时取等。KL 不对称，因此 $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$。

### 4.3 交叉熵、熵与 KL 的关系

$$\underbrace{H(P, Q)}_{\text{交叉熵}} = \underbrace{H(P)}_{\text{熵}} + \underbrace{D_{KL}(P \parallel Q)}_{\text{KL 散度}}$$

即 $-\mathbb{E}_{x \sim P}[\log Q(x)] = H(P) + D_{KL}(P \parallel Q)$。最小化交叉熵等价于最小化 $P$ 与 $Q$ 的 KL 散度。

### 4.4 Jensen–Shannon 散度

$$\text{JSD}(P \parallel Q) = \frac{1}{2}D_{KL}\left(P \parallel M\right) + \frac{1}{2}D_{KL}\left(Q \parallel M\right), \qquad M = \frac{P + Q}{2}$$

JSD 对称且恒非负，是 GAN 原始目标所最小化的量。

---

## 5. 常用恒等式

1. **Softmax 梯度**：设 $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$，则
   $$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$
2. **Log-sum-exp 稳定性**：$\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i - m}$，其中 $m = \max_i z_i$（数值稳定，FlashAttention 的 Online Softmax 依赖此式）。
3. **高斯积分的矩**：$\mathbb{E}_{x\sim\mathcal{N}(\mu,\sigma^2)}[x] = \mu$，$\mathbb{E}[x^2] = \mu^2 + \sigma^2$。
4. **指数族梯度**：`注意：以下公式在 KaTeX 中如需正确渲染，请勿在 `\text` 中混入未转义的中文括号。`

---

## 6. 参考文献

[1] Goodfellow, I., Bengio, Y. & Courville, A. Deep Learning. MIT Press, 2016. (Chapter 2: Linear Algebra, 3: Probability and Information Theory)

[2] Boyd, S. & Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.

[3] Strang, G. Linear Algebra and Its Applications. 4th ed., 2006.

[4] Cover, T. & Thomas, J. Elements of Information Theory. Wiley, 2006.

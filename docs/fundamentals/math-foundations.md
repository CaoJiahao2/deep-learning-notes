# 数学基础

> 深度学习推导所需的线性代数、微积分、概率与信息论基础，以及范数/距离与流形几何视角。后续所有文档的公式都建立在本章符号与恒等式之上。

---

## 目录

1. [线性代数](#1)
2. [微积分与优化](#2)
3. [概率与统计](#3)
4. [信息论](#4)
5. [常用恒等式](#5)
6. [范数与距离](#6)
7. [几何视角：流形假设](#7)
8. [参考文献](#8)

---

## 1. 线性代数

### 1.1 向量与矩阵内积

两个向量 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ 的内积：

$$\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta$$

其中 $\theta$ 是两个向量之间的夹角。内积越大，两向量越"相似"（方向越一致）。

**在深度学习中的用途**：注意力打分（query 与 key 的内积）、embedding 相似度、全连接层的本质运算。

### 1.2 矩阵乘法与向量化

对 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 与 $\mathbf{W} \in \mathbb{R}^{d \times h}$，$\mathbf{Y} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{n \times h}$：

$$Y_{ij} = \sum_{k=1}^{d} X_{ik} W_{kj}$$

**矩阵迹（Trace）的循环性质**（在梯度推导中反复用到）：

$$\operatorname{Tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{Tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \operatorname{Tr}(\mathbf{B}\mathbf{C}\mathbf{A})$$

### 1.3 特征值与奇异值

- **特征值分解**：对称矩阵 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$，$\mathbf{Q}$ 为正交矩阵，$\boldsymbol{\Lambda}$ 为对角特征值矩阵。特征对满足 $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$。
- **奇异值分解（SVD）**：任意矩阵 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，$\boldsymbol{\Sigma}$ 的对角元为奇异值。SVD 是 PCA、低秩近似（LoRA 的理论基础）的核心工具。

两者的桥梁：奇异值是 $\mathbf{A}^\top\mathbf{A}$ 特征值的平方根，$\sigma_i(\mathbf{A}) = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$；最大奇异值给出谱范数 $\|\mathbf{A}\|_2 = \sigma_1$（矩阵对向量的最大拉伸倍数）。

**低秩近似（Eckart–Young 定理）**：对秩为 $r$ 的矩阵，其最佳 $k$ 秩近似（Frobenius 范数意义下）为截断 SVD：

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top, \qquad \min_{\operatorname{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

这是 LoRA 假设"权重更新 $\Delta W$ 是低秩的"的数学依据。

### 1.4 矩阵求导

深度学习反向传播的核心是**标量对矩阵的梯度**。设 $L$ 为标量损失：

- $L = \mathbf{a}^\top \mathbf{x}$，则 $\dfrac{\partial L}{\partial \mathbf{x}} = \mathbf{a}$
- $L = \mathbf{x}^\top \mathbf{A} \mathbf{x}$（$\mathbf{A}$ 对称），则 $\dfrac{\partial L}{\partial \mathbf{x}} = 2\mathbf{A}\mathbf{x}$
- $L = \|\mathbf{x}\|_2^2 = \mathbf{x}^\top \mathbf{x}$，则 $\dfrac{\partial L}{\partial \mathbf{x}} = 2\mathbf{x}$

### 1.5 谱定理与正定矩阵

**谱定理**：任意实对称矩阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 都可以正交对角化 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$，且特征值全为实数。几何上，对称矩阵的作用可以分解为"旋转 → 沿正交轴缩放 → 旋转回来"。

**正定与半正定**：对称矩阵 $\mathbf{A}$ 半正定（记 $\mathbf{A} \succeq 0$）当且仅当对所有 $\mathbf{x} \ne \mathbf{0}$ 有二次型 $\mathbf{x}^\top \mathbf{A} \mathbf{x} \ge 0$，等价于所有特征值 $\lambda_i(\mathbf{A}) \ge 0$，即"任何方向上的二次型都非负" ⇔ "所有特征值非负"。

**在深度学习中的用途**：协方差矩阵、Hessian、核方法中的 Gram 矩阵、图神经网络里的图拉普拉斯 $\mathbf{L} = \mathbf{D} - \mathbf{A}$ 都是对称（半正定）矩阵，谱定理是分析它们的总入口——Hessian 特征值符号给出曲率方向（见 2.6），协方差谱给出主成分（见 1.6），图拉普拉斯的谱是谱图卷积的基础。

### 1.6 SVD 与 PCA 的关系

对去均值数据 $\mathbf{X}_c \in \mathbb{R}^{n \times d}$（每行一个样本，$\bar{\mathbf{x}}$ 为均值）做 SVD：

$$\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \implies \mathbf{C} = \frac{1}{n}\mathbf{X}_c^\top\mathbf{X}_c = \mathbf{V}\left(\frac{1}{n}\boldsymbol{\Sigma}^2\right)\mathbf{V}^\top$$

即**协方差矩阵的特征分解 = 中心化数据的 SVD**：右奇异向量（$\mathbf{V}$ 的列）就是主成分方向，奇异值平方给出各方向上的方差。投影与降维：

$$\mathbf{z} = \mathbf{V}_k^\top(\mathbf{x} - \bar{\mathbf{x}}) \in \mathbb{R}^k, \qquad \rho_k = \frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_{i=1}^{r}\sigma_i^2}$$

$\rho_k$ 是前 $k$ 个主成分的**方差解释率**。结合 Eckart–Young 定理（1.3 节）：PCA 方向既是最大方差方向，也是最小线性重构误差方向——两个目标在数学上等价。

**在深度学习中的用途**：降维与可视化、输入白化（去相关 + 单位方差）、embedding 空间分析（考察表示的有效维度），以及为 LoRA 的低秩假设提供直觉（权重矩阵的谱通常衰减很快）。

---

## 2. 微积分与优化

### 2.1 链式法则

设 $z = f(y)$，$y = g(x)$，则：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

对向量：$z = f(\mathbf{y})$，$\mathbf{y} = g(\mathbf{x})$，则 Jacobian 形式：

$$\frac{\partial z}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^\top \frac{\partial z}{\partial \mathbf{y}}$$

深度网络是 $u_0 \to u_1 \to \dots \to u_L$ 的复合函数，反向传播把链式法则从输出端逐层回传（向量-Jacobian 积，见 2.5 节）。关键性质：**一次反向扫描同时得到关于所有参数的梯度**，代价与一次前向计算同阶——这是反向传播相对数值差分（每个参数一次前向）的根本优势。

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

### 2.5 Jacobian

向量函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的 Jacobian 是全部一阶偏导组成的矩阵：$\mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \dfrac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$，其中 $\left[\mathbf{J}\right]_{ij} = \partial f_i / \partial x_j$。

复合函数的 Jacobian 是链式法则的矩阵形式：$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \mathbf{J}_{\mathbf{g}}$。反向传播实际计算的是**向量-Jacobian 积（VJP）**$\mathbf{v}^\top \mathbf{J}$——只把上游梯度"穿过"当前层，从不显式构造整个 $\mathbf{J}$。

**在深度学习中的用途**：VJP 是自动微分的实现语言；normalizing flows 需要每层变换的 $\log\left|\det \mathbf{J}_f\right|$（变换的体积变化）；分析 ReLU 网络的逐段线性结构也依赖 Jacobian。

### 2.6 Hessian 与曲率

二阶偏导组成的对称矩阵：

$$\mathbf{H} = \nabla^2 f(\mathbf{x}) \in \mathbb{R}^{n \times n}, \qquad H_{ij} = \frac{\partial^2 f}{\partial x_i\, \partial x_j}$$

由谱定理（1.5 节），$\mathbf{H}$ 的特征值符号描述各方向曲率：全正 = 局部凸碗，全负 = 倒扣碗，混合 = 鞍点。梯度的 Lipschitz 常数由最大特征值控制：$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|_2 \le \|\mathbf{H}\|_2\, \|\mathbf{x} - \mathbf{y}\|_2 = L\, \|\mathbf{x} - \mathbf{y}\|_2$。

**在深度学习中的用途**：学习率上限由曲率决定（即 2.3 节的 $L$）；牛顿法方向 $-\mathbf{H}^{-1}\nabla f$ 相当于按曲率预条件的梯度；"平坦/尖锐极小值"（Hessian 谱的形状）与泛化能力的关联是损失景观研究的常用分析工具。

### 2.7 反向传播实例：两层 MLP

前向计算：$\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1$，$\mathbf{h} = \sigma(\mathbf{z}_1)$，$\hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$，损失 $L = -\log \hat{y}_k$。反向传播（$\odot$ 为逐元素乘）：

$$\boldsymbol{\delta}_2 = \hat{\mathbf{y}} - \mathbf{y} \quad \Rightarrow \quad \nabla_{\mathbf{W}_2} L = \boldsymbol{\delta}_2 \mathbf{h}^\top, \qquad \nabla_{\mathbf{b}_2} L = \boldsymbol{\delta}_2$$

$$\boldsymbol{\delta}_1 = \left(\mathbf{W}_2^\top \boldsymbol{\delta}_2\right) \odot \sigma'(\mathbf{z}_1) \quad \Rightarrow \quad \nabla_{\mathbf{W}_1} L = \boldsymbol{\delta}_1 \mathbf{x}^\top, \qquad \nabla_{\mathbf{b}_1} L = \boldsymbol{\delta}_1$$

两个普适规律：**误差信号 $\boldsymbol{\delta}$ 逐层回传，每层梯度 = 该层输入 × 上游误差**；softmax + 交叉熵组合使输出层误差恰为 $\hat{\mathbf{y}} - \mathbf{y}$（第 5 节恒等式 1），无需显式处理 softmax 的二阶项。完整推导见 [反向传播](../training/backpropagation.md)。

---

## 3. 概率与统计

### 3.1 条件概率与贝叶斯定理

$$P(A \mid B) = \frac{P(A, B)}{P(B)}, \qquad P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

**条件独立** $X \perp Y \mid Z$（给定 $Z$ 后 $X$ 与 $Y$ 独立）是朴素贝叶斯、马尔可夫性质与图模型的基础。贝叶斯定理的"深度学习读法"：后验 $\propto$ 似然 × 先验——MAP 估计把正则项解释为先验（L2 ↔ 高斯先验，L1 ↔ 拉普拉斯先验）。

### 3.2 期望与方差

$$\mathbb{E}[X] = \sum_x x P(x), \qquad \operatorname{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2$$

**方差分解（Law of Total Variance）**：

$$\operatorname{Var}[X] = \mathbb{E}[\operatorname{Var}[X \mid Y]] + \operatorname{Var}[\mathbb{E}[X \mid Y]]$$

独立随机变量之和的方差可加：$X \perp Y \implies \operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$。

**在深度学习中的用途**：mini-batch 梯度是全量梯度的无偏估计，其方差随 batch size 增大而线性下降——这是"学习率可随 batch size 放大"的统计依据。

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

### 3.5 常见分布

| 分布 | PMF / PDF | 在深度学习中的角色 |
|---|---|---|
| Bernoulli | $p^x (1-p)^{1-x}$，$x \in \{0, 1\}$ | 二分类：sigmoid 输出 + BCE |
| Categorical | $\prod_c \theta_c^{x_c}$，$\mathbf{x}$ 为 one-hot | 多分类：softmax 输出 + CE、语言建模 |
| Gaussian | 见 3.4 节 | 回归 MSE、VAE、扩散模型、权重初始化 |
| Laplace | $p(x) = \frac{1}{2b}\exp\left(-\frac{|x - \mu|}{b}\right)$ | L1 损失 = Laplace 噪声下的 MLE；稀疏先验 |
| Uniform | $p(x) = \frac{1}{b-a}$，$x \in [a, b]$ | Xavier / He 初始化采样区间、数据增强 |

一个统一的视角：**换损失函数 = 隐式换噪声假设**。MSE 对应高斯噪声下的 MLE，L1 对应拉普拉斯噪声下的 MLE，交叉熵对应分类分布下的 MLE。

### 3.6 协方差矩阵与多元高斯的几何

协方差矩阵 $\boldsymbol{\Sigma} = \mathbb{E}\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\right] \succeq 0$。由谱定理 $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$：多元高斯的等概率密度线是以 $\boldsymbol{\mu}$ 为中心的椭球，主轴方向是特征向量、相对轴长是 $\sqrt{\lambda_i}$。白化变换把它变成各向同性的标准高斯：

$$\mathbf{z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu}) \implies \operatorname{Cov}[\mathbf{z}] = \mathbf{I}$$

**在深度学习中的用途**：BatchNorm / whitening 的目标就是把各通道的协方差拉向各向同性（消除特征间相关与量纲差异）；度量学习中的 Mahalanobis 距离（见 6.3 节）就是用 $\boldsymbol{\Sigma}^{-1}$ 加权的距离。

---

## 4. 信息论

### 4.1 熵、联合熵与条件熵

$$H(X) = -\sum_x p(x) \log p(x), \qquad H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

$$H(Y \mid X) = -\sum_{x,y} p(x,y) \log p(y \mid x)$$

### 4.2 KL 散度

$$D_{KL}(P \parallel Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right] = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

性质：$D_{KL}(P \parallel Q) \ge 0$，当且仅当 $P = Q$ 时取等。KL 不对称，因此 $D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)$。

**在深度学习中的用途**：知识蒸馏最小化温度软化后的师生分布 KL；RLHF / DPO 用 KL 惩罚约束策略不偏离参考模型；VAE 用 KL 正则把后验拉向先验。方向也有讲究：forward KL 偏"均值搜索"（覆盖所有模式），reverse KL 偏"模式寻找"（可能丢弃模式）。

### 4.3 交叉熵、熵与 KL 的关系

$$\underbrace{H(P, Q)}_{\text{交叉熵}} = \underbrace{H(P)}_{\text{熵}} + \underbrace{D_{KL}(P \parallel Q)}_{\text{KL 散度}}$$

即 $-\mathbb{E}_{x \sim P}[\log Q(x)] = H(P) + D_{KL}(P \parallel Q)$。最小化交叉熵等价于最小化 $P$ 与 $Q$ 的 KL 散度。

### 4.4 Jensen–Shannon 散度

$$\text{JSD}(P \parallel Q) = \frac{1}{2}D_{KL}\left(P \parallel M\right) + \frac{1}{2}D_{KL}\left(Q \parallel M\right), \qquad M = \frac{P + Q}{2}$$

JSD 对称且恒非负，是 GAN 原始目标所最小化的量。

### 4.5 互信息

$$I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)} = D_{KL}\left(p(x, y) \,\|\, p(x)\, p(y)\right) \ge 0$$

互信息 = 联合分布与"独立假设分布"的 KL 散度，衡量"知道 $Y$ 之后 $X$ 的不确定性减少了多少"。等价形式：$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$；且 $I(X; Y) = 0$ 当且仅当 $X$ 与 $Y$ 独立。

**在深度学习中的用途**：对比学习（InfoNCE / CLIP）的损失是互信息的下界估计——拉近正样本对、推远负样本对，等价于最大化两个视图之间的互信息；信息瓶颈把"好表示"形式化为"对输入尽量压缩、对目标尽量保信息"；特征选择与表征解耦也以互信息为基本量度。

### 4.6 深度学习中的信息论损失一览

| 损失 | 信息论量 | 典型场景 |
|---|---|---|
| 交叉熵 $\mathcal{L}_{\text{CE}}$ | $H(P, Q)$ | 分类、语言建模（MLE） |
| KL 散度 | $D_{KL}(P \parallel Q)$ | 知识蒸馏、RLHF 正则、VAE 先验 |
| JS 散度 | JSD | GAN 目标 |
| InfoNCE | $I(X; Y)$ 的下界 | CLIP、CPC 等对比学习 |

VAE 的 ELBO 把"重构"与"压缩"显式拆开：$\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{KL}\left(q_\phi(z \mid x)\, \|\, p(z)\right)$，第一项是重构项，第二项是 KL 正则。

统一视角：**许多训练目标都是"让模型分布接近数据分布"**，差别只在度量（KL / JSD / 互信息下界）与方向（forward / reverse KL）。

---

## 5. 常用恒等式

1. **Softmax 梯度**：设 $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$，则
   $$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$
2. **Log-sum-exp 稳定性**：$\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i - m}$，其中 $m = \max_i z_i$（数值稳定，FlashAttention 的 Online Softmax 依赖此式）。
3. **高斯积分的矩**：$\mathbb{E}_{x\sim\mathcal{N}(\mu,\sigma^2)}[x] = \mu$，$\mathbb{E}[x^2] = \mu^2 + \sigma^2$。
4. **指数族梯度**：`注意：以下公式在 KaTeX 中如需正确渲染，请勿在 \text 中混入未转义的中文括号。`
5. **高斯微分熵**：$H\left(\mathcal{N}(\mu, \sigma^2)\right) = \dfrac{1}{2}\log\left(2\pi e \sigma^2\right)$——给定方差下，高斯是熵最大的分布。
6. **两个高斯之间的 KL 散度**：
   $$D_{KL}\left(\mathcal{N}(\mu_1, \sigma_1^2)\, \|\, \mathcal{N}(\mu_2, \sigma_2^2)\right) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$
   VAE 的 KL 正则项（对角高斯后验 vs 标准高斯先验）就是它在各维上的求和。
7. **独立维度 KL 的可加性**：$D_{KL}\left(\prod_i p_i \,\|\, \prod_i q_i\right) = \sum_i D_{KL}(p_i \,\|\, q_i)$。

---

## 6. 范数与距离

### 6.1 向量范数与矩阵范数

$p$-范数定义为 $\|\mathbf{x}\|_p = \left(\sum_{i=1}^{d} |x_i|^p\right)^{1/p}$（$p \ge 1$），另有 $\|\mathbf{x}\|_\infty = \max_i |x_i|$。常用的三个：$p = 1$（单位球是菱形，尖点在坐标轴上）、$p = 2$（圆球）、$p = \infty$（方形）。矩阵范数最常用 Frobenius 范数 $\|\mathbf{A}\|_F$（把矩阵当长向量算 $L_2$）与谱范数 $\|\mathbf{A}\|_2 = \sigma_1(\mathbf{A})$（最大拉伸倍数）。有限维空间中所有范数等价（只差常数倍），但**数值与几何后果差异巨大**——用哪个范数做正则化会直接改变解的性质。

### 6.2 范数正则化

在经验风险之后加上范数惩罚：$\lambda\|\mathbf{w}\|_2^2$ 是 ridge / weight decay，$\lambda\|\mathbf{w}\|_1$ 是 lasso。

- **$L_2$（ridge / weight decay）**：把权重整体往原点收缩，对特征共线更稳定，但不产生精确的零；
- **$L_1$（lasso）**：约束球的尖点落在坐标轴上，最优解容易停在轴上 ⇒ 部分权重恰为 0，产生**稀疏性**；
- 同族约束：梯度裁剪 $\|\mathbf{g}\|_2 \le c$（RNN/Transformer 训练标配）、谱归一化 $\|\mathbf{W}\|_2 \le 1$（SN-GAN 稳定判别器）。

### 6.3 距离与相似度

- **余弦相似度**：$\cos(\mathbf{a}, \mathbf{b}) = \dfrac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$——embedding 检索的标准度量，只看方向、对模长不敏感；
- **欧氏距离**：即 $L_2$ 距离，对应 MSE 与高斯噪声假设；
- **Mahalanobis 距离**：$d_{\boldsymbol{\Sigma}}(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \mathbf{y})}$——按协方差"拉直"空间后的距离，$\boldsymbol{\Sigma} = \mathbf{I}$ 时退化为欧氏距离；
- **Wasserstein 距离**：把一个概率分布"搬运"到另一个的最小代价，是 WGAN 与分布比较的核心工具。

**在深度学习中的用途**：检索（embedding 相似度）、聚类（k-means 用欧氏）、异常检测（Mahalanobis 距离做 OOD 检测）、度量学习（学一个任务专用的马氏距离）。

---

## 7. 几何视角：流形假设

**流形假设**：真实高维数据集中在嵌入于 $\mathbb{R}^d$ 中的低维流形 $\mathcal{M}^k$（$k \ll d$）附近。例如 $512 \times 512$ 的图像像素空间是 $\mathbb{R}^{262144}$，但"自然图像"只占其中极小且高度结构化的一块——其内在维度（物体类别、位姿、光照等因子数）远小于像素数。

**维度灾难**：高维空间中体积向球壳集中、随机点对之间的距离趋于同值（最近与最远邻居的距离比趋近 1），基于"均匀分布"与"距离"的直觉全部失效。流形假设是"为什么深度学习在高维数据上仍然可行"的主要解释：有效数据维度其实是 $k$ 而不是 $d$。

**在深度学习中的用途**：

1. **表示学习 ≈ 学流形坐标**：编码器把 $\mathbf{x} \in \mathbb{R}^d$ 映射为低维潜码 $\mathbf{z} \in \mathbb{R}^k$，好的表示就是流形的参数化（PCA 是线性版本，autoencoder 是非线性版本）；
2. **生成建模 ≈ 在流形上造分布**：GAN / 扩散 / flow 的本质困难都是"把简单分布搬运到弯曲的低维流形上"；
3. **数据增强 ≈ 沿流形切向的扰动**：有效增强（裁剪、色调变换）大致沿流形移动样本，而各向同性随机噪声会把样本推离流形（无效甚至有害）；
4. **分类边界**：原始空间中线性不可分的类别，在多层网络"折叠/展开"后的特征空间中可能轻易线性可分。

可视化工具 t-SNE / UMAP 的目标即"把高维流形嵌入二维并保持邻域结构"，是检查学到的表示质量的常用手段。

---

## 8. 参考文献

[1] Goodfellow, I., Bengio, Y. & Courville, A. Deep Learning. MIT Press, 2016. (Chapter 2: Linear Algebra, 3: Probability and Information Theory)

[2] Boyd, S. & Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.

[3] Strang, G. Linear Algebra and Its Applications. 4th ed., 2006.

[4] Cover, T. & Thomas, J. Elements of Information Theory. Wiley, 2006.

[5] Shannon, C. A Mathematical Theory of Communication. Bell System Technical Journal, 1948.

[6] Bishop, C. Pattern Recognition and Machine Learning. Springer, 2006.

[7] Eckart, C. & Young, G. The Approximation of One Matrix by Another of Lower Rank. Psychometrika, 1936.

[8] Kingma, D. & Welling, M. Auto-Encoding Variational Bayes. ICLR, 2014.

[9] Hinton, G., Vinyals, O. & Dean, J. Distilling the Knowledge in a Neural Network. 2015. arXiv:1503.02531.

[10] van den Oord, A., Li, Y. & Vinyals, O. Representation Learning with Contrastive Predictive Coding. 2018. arXiv:1807.03748.

[11] van der Maaten, L. & Hinton, G. Visualizing Data using t-SNE. JMLR, 2008.

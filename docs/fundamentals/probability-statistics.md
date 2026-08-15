# 概率论与统计深入

> 从随机变量与分布族到估计与假设检验 —— 深度学习不确定性、生成模型与对齐方法的数学基础。

---

## 目录

1. [概述](#1)
2. [随机变量与分布族](#2)
3. [期望、方差与矩](#3)
4. [联合分布与条件分布](#4)
5. [极限定理](#5)
6. [统计估计：MLE、MAP 与贝叶斯](#6)
7. [假设检验与置信区间](#7)
8. [深度学习中的应用](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

本章深化 `math-foundations.md` 中的概率与统计内容，从基础定义走向估计、检验与极限定理，为理解不确定性量化、生成模型、RLHF/DPO 等提供数学支撑。

核心问题：**给定观测数据，如何建模其不确定性并做出推断？**

---

## 2 随机变量与分布族 {#2}

### 2.1 离散与连续

- **离散**：取值可数，用概率质量函数 $P(X = x)$；
- **连续**：取值连续，用概率密度函数 $p(x)$。

### 2.2 常见离散分布

| 分布 | 参数 | 含义 | PMF |
|------|------|------|-----|
| Bernoulli | $p$ | 一次伯努利试验 | $p^x (1-p)^{1-x}$ |
| Binomial | $n, p$ | $n$ 次独立试验成功次数 | $\binom{n}{k}p^k(1-p)^{n-k}$ |
| Poisson | $\lambda$ | 单位时间事件数 | $\frac{\lambda^k e^{-\lambda}}{k!}$ |
| Categorical | $\mathbf{p}$ | 多类 | $\prod_c p_c^{\mathbb{I}(y=c)}$ |

### 2.3 常见连续分布

| 分布 | 参数 | 密度（示意） |
|------|------|-------------|
| 正态 | $\mu, \sigma^2$ | $p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| 均匀 | $a, b$ | $p(x)=\frac{1}{b-a}$ |
| 指数 | $\lambda$ | $p(x)=\lambda e^{-\lambda x}$ |
| Beta | $\alpha, \beta$ | 伯努利参数先验 |

### 2.4 指数族

许多分布属于指数族：

$$p(x \mid \theta) = h(x) \exp\left(\eta(\theta)^\top T(x) - A(\theta)\right)$$

其中 $T(x)$ 为充分统计量，$A(\theta)$ 为对数配分函数。正态、Bernoulli、Poisson 均属此类，是广义线性模型的基础。

---

## 3 期望、方差与矩 {#3}

### 3.1 期望

$$\mathbb{E}[X] = \sum_x x P(x) \quad \text{或} \quad \int x p(x) dx$$

线性性：$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$。

### 3.2 方差与协方差

$$\operatorname{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2$$

$$\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$$

$$\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y] + 2\operatorname{Cov}(X, Y)$$

### 3.3 矩生成函数

$$M_X(t) = \mathbb{E}[e^{tX}]$$

矩通过 $M_X^{(k)}(0) = \mathbb{E}[X^k]$ 得到，常用于推导分布性质。

---

## 4 联合分布与条件分布 {#4}

### 4.1 贝叶斯定理

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

深度学习中的常见形式：

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})}$$

- 后验 $\propto$ 似然 × 先验；
- $p(\mathcal{D}) = \int p(\mathcal{D}\mid\theta)p(\theta)d\theta$ 为证据（marginal likelihood）。

### 4.2 独立性

$X \perp Y$ 当且仅当 $P(X,Y) = P(X)P(Y)$。条件独立 $X \perp Y \mid Z$ 当且仅当 $P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$。

### 4.3 全概率与全方差

$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

$$\operatorname{Var}[X] = \mathbb{E}[\operatorname{Var}[X\mid Y]] + \operatorname{Var}[\mathbb{E}[X\mid Y]]$$

### 4.4 马尔可夫性质

$X_t$ 的未来只依赖当前状态：$P(X_{t+1} \mid X_t, X_{t-1}, \ldots) = P(X_{t+1} \mid X_t)$。这是 MDP/RL（见 `reinforcement-learning.md`）与序列建模的基础。

---

## 5 极限定理 {#5}

### 5.1 大数定律（LLN）

样本均值依概率收敛到期望：

$$\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mathbb{E}[X]$$

意义：独立同分布样本足够多时，经验均值接近真实均值（训练中 batch 梯度估计的保证）。

### 5.2 中心极限定理（CLT）

独立同分布样本和的标准化趋于正态：

$$\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$$

意义：大量随机因素叠加趋于正态，是噪声建模与置信区间的依据。

### 5.3 重要性

- LLN 保证蒙特卡洛估计（梯度、期望、dropout 平均）收敛；
- CLT 支撑假设检验与置信区间。

---

## 6 统计估计：MLE、MAP 与贝叶斯 {#6}

### 6.1 最大似然估计（MLE）

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p_\theta(x_i)$$

等价于最小化经验分布的交叉熵。深度学习分类损失（cross-entropy）本质是 MLE。

### 6.2 最大后验估计（MAP）

加入先验：

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\sum_i \log p_\theta(x_i) + \log p(\theta)\right]$$

- 高斯先验 $\theta \sim \mathcal{N}(0, \sigma^2 I)$ 对应 L2 正则化（weight decay）；
- Laplace 先验对应 L1 正则化（稀疏）。

### 6.3 贝叶斯估计

不取点估计，而是计算完整后验：

$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)$$

- 预测用后验积分：$p(y \mid x, \mathcal{D}) = \int p(y\mid x, \theta)p(\theta\mid\mathcal{D})d\theta$；
- 后验通常不可解析，需采样（MCMC、变分推断）。

### 6.4 偏差-方差权衡

期望泛化误差分解为：

$$\underbrace{\mathbb{E}[(y - \hat{f}(x))^2]}_{\text{期望误差}} = \underbrace{\text{Bias}^2}_{\text{偏差}} + \underbrace{\text{Var}}_{\text{方差}} + \underbrace{\sigma^2}_{\text{噪声}}$$

复杂模型低偏差高方差，简单模型高偏差低方差（见 `machine-learning.md`）。

---

## 7 假设检验与置信区间 {#7}

### 7.1 假设检验框架

- 原假设 $H_0$ 与备择假设 $H_1$；
- $p$ 值：在原假设下观察到的结果（或更极端）的概率；
- 显著性水平 $\alpha$（常用 0.05）：$p < \alpha$ 拒绝 $H_0$；
- **两类错误**：I 型（假阳性）与 II 型（假阴性）。

### 7.2 常见检验

- $t$ 检验：均值差异；
- 卡方检验：分类变量独立性；
- ANOVA：多组均值差异；
- 正态性检验：Shapiro-Wilk。

### 7.3 置信区间

参数 $\theta$ 的 $95\%$ 置信区间是随机区间，覆盖真值的概率为 $0.95$：

$$\bar{x} \pm z_{0.975} \frac{\sigma}{\sqrt{n}}$$

### 7.4 与 AI 的关系

- 模型对比需做显著性检验（A/B 测试，见 `llm-ops.md`）；
- 报告评估结果时应含置信区间，而非单一分数。

---

## 8 深度学习中的应用 {#8}

| 概念 | 应用 |
|------|------|
| 交叉熵 / MLE | 分类与语言模型损失 |
| 重参数化技巧 | VAE、扩散模型 |
| KL / JSD | 对齐、蒸馏、生成目标 |
| 贝叶斯推断 | 不确定性量化、贝叶斯网络 |
| 期望 | 期望梯度（policy gradient，见 `reinforcement-learning.md`） |
| 条件分布 | 自回归模型 $P(x_{t+1}\mid x_{\le t})$ |
| 高斯噪声 | 扩散模型加噪、初始化 |

---

## 9 实践建议 {#9}

- 先掌握分布、期望、贝叶斯与极限定理，再学估计与检验；
- 理解"交叉熵 = 负对数似然"的连接，是深度学习损失设计的钥匙；
- 正则化（weight decay）从 MAP 视角理解更深刻（见 `training/regularization.md`）；
- 模型评估务必报告不确定性（方差/置信区间）；
- 结合 `math-foundations.md` 与本篇，形成完整的数学地基。

---

## 10 参考文献 {#10}

[1] C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.

[2] K. P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.

[3] G. Casella, R. L. Berger, *Statistical Inference*, Cengage.

[4] D. Barber, *Bayesian Reasoning and Machine Learning*, Cambridge University Press.

[5] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.

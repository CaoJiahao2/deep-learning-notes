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

### 2.5 分布族速查表

| 分布 | PMF / PDF | 均值 | 方差 | 深度学习角色 |
|------|-----------|------|------|--------------|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | 二分类标签、dropout 掩码 |
| Categorical($\mathbf{p}$) | $\prod_c p_c^{\mathbb{I}(y=c)}$ | — | — | softmax 输出、词表采样 |
| Binomial($n, p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | k/n 成功率的抽样分布 |
| Poisson($\lambda$) | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | 稀疏计数、事件流建模 |
| Normal($\mu, \sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | 噪声、初始化、diffusion |
| Uniform($a, b$) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 权重初始化、增强采样 |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | 事件间隔（无记忆性） |
| Laplace($\mu, b$) | $\frac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | L1 正则的先验、鲁棒回归 |
| Beta($\alpha, \beta$) | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha,\beta)}$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | 概率参数先验、Thompson 采样 |
| Dirichlet($\boldsymbol{\alpha}$) | $\propto \prod_c x_c^{\alpha_c-1}$ | $\alpha_c/\alpha_0$ | — | 类分布先验、主题模型 |

其中 $\mathrm{B}(\cdot,\cdot)$ 为 Beta 函数，$\alpha_0 = \sum_c \alpha_c$，"—" 表示无简洁标量表达。表中绝大多数成员都属于 §2.4 的指数族；它们作为先验的共轭关系见 §6.6。

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

### 4.5 相关与因果：混杂变量与 do-演算直觉

- **相关 ≠ 因果**：$\operatorname{Cov}(X, Y) \neq 0$ 只说明二者在观测分布中共变；"干预 $X$ 会改变 $Y$"是关于干预分布的断言，两者可以完全相反。Simpson 悖论是极端体现：分组结论与汇总结论方向相反——"该看哪张表"本身由因果假设决定。
- **混杂变量（confounder）**：同时影响 $X$ 与 $Y$ 的公共原因 $Z$ 会打开一条非因果路径。经典例子：冰淇淋销量与溺水人数正相关，混杂变量是气温。
- 用因果图（DAG）表达假设后，观测条件与干预的区别可写成：

$$\underbrace{P(Y \mid X = x)}_{\text{观测}} \;\neq\; \underbrace{P(Y \mid do(X = x))}_{\text{干预}} \quad (\text{一般不等})$$

- **后门调整的直觉**：若变量集 $Z$ 阻断了 $X$ 到 $Y$ 的所有"后门"路径，干预分布可由观测分布还原：

$$P(Y \mid do(X = x)) = \sum_z P(Y \mid X = x, Z = z)\, P(Z = z)$$

即"在每个 $Z$ 层内估计效应，再按 $P(z)$ 加权平均"。随机对照试验（RCT）是金标准，正因为随机化切断了所有后门路径。
- **do-演算**：Pearl 给出的三条重写规则刻画 $do(\cdot)$ 与条件化何时可互换，是因果推断的完备演算系统。深度学习中的对应物：off-policy 评估、不变风险最小化、反事实数据增强，处理的都是干预/反事实分布而非纯观测分布。

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

### 5.4 CLT 在初始化与归一化中的体现

"大量独立项之和的方差可加"是 CLT 的核心直觉，初始化的方差分析正是它的直接应用。对线性层 $z_i = \sum_j W_{ij} x_j$，设 $W_{ij}$ 零均值、方差 $\sigma_w^2$、与输入独立：

$$\operatorname{Var}[z_i] = \sum_{j=1}^{n_{\mathrm{in}}} \mathbb{E}[W_{ij}^2]\,\mathbb{E}[x_j^2] = n_{\mathrm{in}}\, \sigma_w^2\, \mathbb{E}[x_j^2]$$

要让每层输出方差 ≈ 输入方差（信号不逐层放大/缩小），需 $\sigma_w^2 = 1/n_{\mathrm{in}}$（前向稳定）；反向传播对上一层提出 $\sigma_w^2 = 1/n_{\mathrm{out}}$。Xavier/LeCun 初始化取折中，常用均匀分布形式：

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},\ \sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right)$$

（方差 $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$ 恰为两个约束的调和平均，配 tanh/sigmoid）。ReLU 负半轴置零使激活方差减半，He/Kaiming 初始化相应取 $\sigma_w^2 = 2/n_{\mathrm{in}}$。

归一化是同一逻辑的运行时版本：BN/LN/RMSNorm 每层把激活拉回标准尺度——不论初始化多精确，乘法结构的方差都会沿深度累积漂移，归一化相当于每层手动重置这一漂移，从而允许更大学习率与更深网络。此外，SGD 的 mini-batch 梯度是 $B$ 个独立样本梯度之和，CLT 说明其围绕真值近似高斯——训练动力学分析中把 batch 噪声建模为高斯，理论依据正在于此。

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

### 6.5 三种估计方式对比

| 维度 | MLE | MAP | 贝叶斯推断 |
|------|-----|-----|-----------|
| 输出 | 点估计 $\hat{\theta}$ | 点估计 $\hat{\theta}$ | 完整后验 $p(\theta \mid \mathcal{D})$ |
| 先验 | 无 | 有（进入点估计） | 有（完整保留） |
| 不确定性 | 无（需渐近近似） | 无 | 内生（后验方差 / credible interval） |
| 计算 | 一次优化 | 一次优化 | 积分 / MCMC / 变分 |
| 大样本极限 | — | $\to$ MLE（先验被冲淡） | 后验集中于 MLE 附近 |
| 深度学习对应 | 交叉熵训练 | weight decay | MC dropout、deep ensembles（近似） |

MLE 的渐近理论是 CLT 的直接推论：$\hat{\theta}_{\text{MLE}} \approx \mathcal{N}(\theta,\, \mathcal{I}^{-1}/N)$，其中 Fisher 信息 $\mathcal{I}$ 刻画参数的可辨识精度——数据量翻倍，估计的不确定性（协方差）减半。

### 6.6 共轭先验

若先验与似然共轭，后验有闭式（与先验同族），贝叶斯更新退化为"数数"：

| 似然 | 共轭先验 | 后验参数 |
|------|----------|----------|
| Bernoulli($p$) | Beta($\alpha, \beta$) | Beta($\alpha + k,\ \beta + n - k$) |
| Categorical($\mathbf{p}$) | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \mathbf{n}$) |
| Gaussian($\mu$，已知 $\sigma^2$) | Normal($\mu_0, \tau^2$) | 精度加权合并 |
| Poisson($\lambda$) | Gamma($\alpha, \beta$) | Gamma($\alpha + n\bar{x},\ \beta + n$) |

两个深度学习中的直接投影：

- **Laplace 平滑 = Beta/Dirichlet 共轭**：先验 Beta(1,1)（均匀），见 $n$ 次中 $k$ 次成功，后验均值为 $\frac{k+1}{n+2}$——语言模型 add-one 平滑的本质是 Dirichlet-Categorical 的后验预测；
- **Gaussian 共轭 = 精度加权平均**：后验均值是先验均值与样本均值按精度（方差倒数）的加权：$\frac{\mu_0/\tau^2 + n\bar{x}/\sigma^2}{1/\tau^2 + n/\sigma^2}$，数据越多权重越偏向数据——先验被"冲淡"，与 §6.5 的收敛关系一致。

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

### 7.5 多重比较

同时检验 $m$ 个假设、每个显著水平 $\alpha = 0.05$，则至少一个假阳性的概率为 $1 - 0.95^m$：$m = 10$ 时已约 $40\%$。深度学习评测中极常见——网格搜索挑"最好"超参、几十个 benchmark 上挑"最好"模型，本质都在做隐式多重比较，选出的往往是运气最好的噪声。校正方法：

- **Bonferroni**：单检验水平收紧为 $\alpha/m$，控制 FWER（至少一个假阳性的概率），简单但保守；
- **Benjamini-Hochberg**：控制 FDR（假发现比例的期望），对大规模筛选（消融实验、神经科学、基因）更合适；
- 更根本的对策：预先注册假设、留出独立验证集做一次终测。

### 7.6 Bootstrap

不依赖渐近分布的不确定性估计：把经验分布当作总体，有放回重采样 $B$ 份大小为 $n$ 的样本，每份算一次统计量，用其分布近似统计量的抽样分布。

```python
import numpy as np

def bootstrap_ci(scores, B=10000, alpha=0.05, seed=0):
    """基准分数均值的 percentile bootstrap 置信区间。"""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    means = rng.choice(scores, size=(B, scores.size), replace=True).mean(axis=1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])
```

适用场景：统计量无解析标准误（中位数、AUC、pass@k、基准总分），或样本量小、正态性可疑。比较两个模型时用 **paired bootstrap**：对同一测试集重采样，考察两者分数差的区间是否含 0。局限：完全依赖观测样本的代表性，外推到分布尾部不可靠。

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

### 8.1 基准评测的统计实践

- **报告区间而非单点**：同一模型多次评测（随机解码、few-shot 采样、工具环境差异）分数会波动，用 CLT 区间 $\bar{x} \pm 1.96\, s/\sqrt{n}$ 或 bootstrap 区间报告"均值 ± 区间"；
- **配对比较**：两个模型应在**同一**测试集上配对比较（同一批样本的逐题对错、差值的区间），而不是比较两个各自带噪声的均值；
- **小测试集陷阱**：$n$ 题准确率的标准差量级为 $\sqrt{p(1-p)/n}$，例如 $p = 0.7$、$n = 500$ 时约 $2\%$——1 个百分点的"提升"很可能只是噪声；
- **固定可固定的随机性**：temperature、seed、数据顺序、工具版本都显式记录；不可固定的（环境非确定性）应报告多次运行方差。

---

## 9 实践建议 {#9}

- 先掌握分布、期望、贝叶斯与极限定理，再学估计与检验；
- 理解"交叉熵 = 负对数似然"的连接，是深度学习损失设计的钥匙；
- 正则化（weight decay）从 MAP 视角理解更深刻（见 `training/regularization.md`）；
- 模型评估务必报告不确定性（方差/置信区间），小样本场景优先 paired bootstrap；
- 把相关性结论升级为因果结论前，先自问"是否控制了混杂变量、能否做干预实验"；
- 结合 `math-foundations.md` 与本篇，形成完整的数学地基。

---

## 10 参考文献 {#10}

[1] C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.

[2] K. P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.

[3] G. Casella, R. L. Berger, *Statistical Inference*, Cengage.

[4] D. Barber, *Bayesian Reasoning and Machine Learning*, Cambridge University Press.

[5] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.

[6] B. Efron & R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[7] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, 2009.

[8] X. Glorot & Y. Bengio, *Understanding the Difficulty of Training Deep Feedforward Neural Networks*, AISTATS, 2010.

[9] K. He, X. Zhang, S. Ren & J. Sun, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, ICCV, 2015.

[10] Y. Benjamini & Y. Hochberg, *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*, JRSS-B, 1995.

# 优化理论与方法

> 从梯度下降收敛到凸优化与 KKT，从自适应优化器到二阶方法 —— 深度学习训练背后的数学。

---

## 目录

1. [概述](#1)
2. [问题设定与分类](#2)
3. [梯度下降与收敛分析](#3)
4. [凸优化与 KKT](#4)
5. [随机梯度下降](#5)
6. [自适应优化器](#6)
7. [二阶方法](#7)
8. [约束与正则化优化](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

深度学习训练的本质是最小化损失函数：

$$\min_{\theta} \ \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

本章提供优化的数学视角，衔接 `math-foundations.md`（凸性、梯度）与 `training/optimization.md`（SGD→AdamW→Lion 的实践演进）。理解优化理论有助于诊断训练不收敛、选择学习率、理解正则化。

---

## 2 问题设定与分类 {#2}

### 2.1 凸 vs 非凸

- **凸函数**：任意两点间线段在函数图像上方，局部最优 = 全局最优；
- **非凸**：存在多个局部最优，深度学习损失高度非凸。

### 2.2 光滑与强凸

- **L-光滑**：梯度 Lipschitz，$\|\nabla f(x) - \nabla f(y)\| \le L\|x-y\|$；
- **μ-强凸**：$f(y) \ge f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$。

光滑与强凸刻画了损失曲面的"形状"，决定收敛速度。

### 2.3 无约束 vs 有约束

- 无约束：$\min f(x)$；
- 有约束：$\min f(x) \ \text{s.t.} \ g_i(x) \le 0, \ h_j(x) = 0$。

---

## 3 梯度下降与收敛分析 {#3}

### 3.1 更新规则

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

### 3.2 光滑函数的收敛

对 $L$-光滑 $f$，取 $\eta = 1/L$：

$$f(x_t) - f(x^*) \le \frac{L\|x_0 - x^*\|^2}{2t}$$

即误差 $O(1/t)$。**梯度下降不要求凸性也能保证梯度趋于 0**（收敛到驻点）。

### 3.3 强凸函数的线性收敛

对 $L$-光滑且 $\mu$-强凸，取最优步长 $\eta = 2/(L+\mu)$：

$$f(x_t) - f(x^*) \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t} \left(f(x_0) - f(x^*)\right)$$

条件数 $\kappa = L/\mu$：$\kappa$ 越小收敛越快。这解释了为什么**预处理（preconditioning）与归一化**能加速收敛。

### 3.4 动量（Momentum）

$$v_{t+1} = \beta v_t + \nabla f(x_t), \quad x_{t+1} = x_t - \eta v_{t+1}$$

动量累积历史梯度，抑制振荡、加速在缓坡方向的移动。

---

## 4 凸优化与 KKT {#4}

### 4.1 凸集与凸函数

凸集：集合内任意两点的连线仍在集合内。凸函数：$f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$。

### 4.2 拉格朗日对偶

对约束问题构造拉格朗日函数：

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

对偶问题给出原问题最优值的下界：$d^* \le p^*$。强对偶成立时 $d^* = p^*$（Slater 条件）。

### 4.3 KKT 条件

凸问题最优解的必要充分条件：

1. **平稳性**：$\nabla_x \mathcal{L} = 0$；
2. **原始可行**：$g_i(x) \le 0, \ h_j(x) = 0$；
3. **对偶可行**：$\lambda_i \ge 0$；
4. **互补松弛**：$\lambda_i g_i(x) = 0$。

应用：SVM 对偶、WGAN 对偶、带约束的优化问题。

### 4.4 内点法与对偶方法

- 内点法：沿中心路径逼近最优；
- 对偶上升/ADMM：分解复杂问题为子问题。

---

## 5 随机梯度下降 {#5}

### 5.1 为什么用随机梯度

全量梯度需要遍历整个数据集，代价高。SGD 用随机小批量估计梯度：

$$\tilde{g} = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell(f_\theta(x_i), y_i), \quad \theta \leftarrow \theta - \eta \tilde{g}$$

由大数定律，$\mathbb{E}[\tilde{g}] = \nabla \mathcal{L}(\theta)$，即**无偏估计**。

### 5.2 收敛分析

对光滑（非凸）函数，SGD 收敛到驻点的期望：

$$\min_{t \le T} \mathbb{E}\|\nabla f(x_t)\|^2 = O\left(\frac{1}{\sqrt{T}}\right)$$

受噪声项 $O(\sigma^2)$ 限制，需通过衰减学习率或批量增大来降低方差。

### 5.3 批量大小与学习率

- 大 batch → 梯度方差小、可加大学习率（线性缩放规则）；
- 小 batch → 噪声大，有正则化效应但收敛不稳；
- batch 与学习率的权衡是训练调参核心（见 `training-pipeline.md`）。

---

## 6 自适应优化器 {#6}

### 6.1 AdaGrad

按参数历史梯度平方和缩放学习率：

$$x_{t+1} = x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

对稀疏参数更新大、对密集参数更新小，但学习率单调递减。

### 6.2 RMSProp

用指数移动平均替代累加，缓解学习率衰减：

$$v_t = \beta v_{t-1} + (1-\beta) g_t^2, \quad x_{t+1} = x_t - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$

### 6.3 Adam

结合动量与自适应步长，是深度学习的默认选择：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$x_{t+1} = x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Adam 的偏差修正使其在训练初期更稳定。

### 6.4 AdamW 与解耦权重衰减

AdamW 把权重衰减从梯度中解耦，直接在参数更新时施加：

$$x_{t+1} = x_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda x_t\right)$$

实践表明 AdamW 的泛化与正则化效果优于 Adam+L2（见 `training/optimization.md`）。

---

## 7 二阶方法 {#7}

### 7.1 牛顿法

用 Hessian 矩阵 $\nabla^2 f(x)$ 做二次近似：

$$x_{t+1} = x_t - \eta \left[\nabla^2 f(x_t)\right]^{-1} \nabla f(x_t)$$

对二次函数一步收敛，但对大规模深度模型 Hessian 太大不可行。

### 7.2 拟牛顿法

用梯度差近似 Hessian 的逆：BFGS、L-BFGS。L-BFGS 只存最近若干梯度差，适合中小规模问题。

### 7.3 自然梯度

用 Fisher 信息矩阵度量参数空间几何：

$$F(\theta) = \mathbb{E}\left[\nabla_\theta \log p_\theta \cdot \nabla_\theta \log p_\theta^\top\right]$$

自然梯度 $F^{-1}\nabla f$ 与 KL 散度相关，是 RL（TRPO）与变分推断的理论基础（见 `reinforcement-learning.md`）。

---

## 8 约束与正则化优化 {#8}

### 8.1 正则化作为约束

- L2（weight decay）：限制参数范数，对应高斯先验；
- L1：诱导稀疏，对应 Laplace 先验；
- Dropout / 数据增强：隐式正则化（见 `training/regularization.md`）。

### 8.2 近端方法

对不可微正则项（如 L1），用近端梯度法：

$$x_{t+1} = \operatorname{prox}_{\eta \lambda \|\cdot\|_1}\left(x_t - \eta \nabla f(x_t)\right)$$

其中 prox 即软阈值算子。

### 8.3 梯度裁剪

防止梯度爆炸（见 `training/training-pipeline.md`）：

$$\tilde{g} = \min\left(1, \frac{c}{\|g\|}\right) g$$

---

## 9 实践建议 {#9}

- 先用 AdamW 起步，再按需调整：稀疏/大模型场景注意权重衰减；
- 理解学习率与 batch 的关系（线性缩放）有助于调参；
- 不收敛时先查梯度消失/爆炸（裁剪、归一化），再调优化器；
- 凸优化与 KKT 是理解 SVM、WGAN、约束建模的基础，值得掌握；
- 结合 `training/optimization.md`（实践演进）与本篇（数学）完整理解优化。

---

## 10 参考文献 {#10}

[1] S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.

[2] D. P. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, ICLR, 2015.

[3] I. Loshchilov, F. Hutter, *Decoupled Weight Decay Regularization (AdamW)*, ICLR, 2019.

[4] L. Bottou, F. E. Curtis, J. Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review, 2018.

[5] J. Nocedal, S. J. Wright, *Numerical Optimization*, Springer, 2006.

[6] Y. Nesterov, *Introductory Lectures on Convex Optimization*, Springer, 2004.

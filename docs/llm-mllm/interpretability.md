# 机制可解释性

> 从稀疏自编码器到 Steering Vectors，打开大模型的"黑盒"。

---

## 目录

1. [概述](#1)
2. [探针方法](#2)
3. [电路分析（Circuits）](#3-circuits)
4. [稀疏自编码器（SAE）](#4-sae)
5. [表征工程与 Steering](#5-steering)
6. [因果追踪与归因](#6)
7. [实践建议](#7)
8. [参考文献](#8)

---

## 1. 概述

机制可解释性（Mechanistic Interpretability）旨在理解神经网络内部"如何"完成计算，而非仅观察输入-输出行为。其核心假设是：大模型内部存在可识别的特征和计算电路（circuits），通过逆向工程可以理解和控制模型行为。

可解释性研究有两个互补的方向：

- **自顶向下（Top-down）**：从行为出发，追踪模型内部哪些组件负责该行为（如因果追踪、探针）。
- **自底向上（Bottom-up）**：从神经元和特征出发，理解每个特征检测什么、如何组合成计算电路（如 SAE、Circuits）。

研究动机包括：安全审计（检测欺骗性对齐）、缺陷定位（幻觉的神经基础）、模型编辑（精确定位并修改知识）、能力验证（确认模型是否真正"理解"）。

---

## 2. 探针方法

### 2.1 线性探针

探针（Probe）是在模型中间层激活上训练的轻量分类器，用于判断特定信息是否在该层被编码：

$$
\hat{y} = \sigma(\mathbf{w}^\top h_i + b)
$$

其中 $h_i$ 是第 $i$ 层的隐藏状态。若探针能准确预测标签 $y$，说明该信息在该层被线性编码。

常见探针应用：

- **词性/句法探针**：检测模型是否编码了语法树结构。
- **知识探针**：判断事实知识存储在哪一层。
- **诚实探针（Honesty Probe）**：检测模型是否知道自己在说谎（Azaria & Mitchell, 2023）。
- **意图探针**：预测模型即将生成有害内容的概率。

### 2.2 探针的注意事项

- 探针本身可能学到任务，而非读取模型已有信息。需用控制任务（control task）排除探针自身能力。
- 线性可分性不等于因果相关性——信息可能被编码但模型实际不使用。
- 探针应在冻结的模型激活上训练，避免与模型联合训练导致信息泄露。

---

## 3. 电路分析（Circuits）

### 3.1 什么是电路

电路（Olah et al., 2020）是神经网络中负责特定功能的子图，由神经元（或注意力头）和它们之间的连接组成。电路分析的目标是找到功能上可解释的最小子图。

典型例子：

- **Induction Heads（Olsson et al., 2022）**：Transformer 中负责"重复前文模式"的注意力头电路，是上下文学习的基础机制。
- **IOI（Indirect Object Identification）电路**：GPT-2-small 中识别间接宾语的 26 个头组成的电路（Wang et al., 2023）。

### 3.2 分析工具

- **激活修补（Activation Patching）**：将输入 A 某位置的激活替换为输入 B 的激活，观察输出变化，定位因果相关组件。
- **路径修补（Path Patching）**：修补特定连接路径而非整个激活，识别信息流路径。
- **消融（Ablation）**：将特定注意力头或 MLP 神经元置零/均值化，观察性能下降。
- **注意力可视化**：检查注意力头的注意力模式，推断其功能。

### 3.3 Induction Heads

Induction head 是 Transformer 中最被充分理解的电路之一，由两个注意力头组成：

1. **前一个 Token 头**：将信息从前一个位置复制到当前位置。
2. **归纳头**：关注当前 Token 之前一次出现后的下一个 Token，实现"重复 A→B 模式"。

Induction heads 的形成与模型的上下文学习能力涌现密切相关，通常在 ~200M 参数规模开始出现。

---

## 4. 稀疏自编码器（SAE）

### 4.1 动机：多义性问题

单个神经元常对多个不相关概念激活（多义性，polysemanticity），例如一个神经元同时激活于"蜘蛛侠"、"英语语法错误"和"狗"。多义性使直接检查神经元含义变得困难。

### 4.2 SAE 原理

稀疏自编码器（Cunningham et al., 2023; Bricken et al., 2023）将模型激活 $h$ 分解为稀疏的、可解释的特征组合：

$$
h \approx \sum_{i=1}^{d_s} z_i \mathbf{d}_i, \quad z_i \geq 0
$$

其中 $\mathbf{d}_i$ 是第 $i$ 个特征方向（字典向量），$z_i$ 是稀疏激活值，$d_s \gg d$（字典维度远大于原始维度，通常 $d_s = 64d$ 或更大）。

SAE 的训练目标：

$$
\mathcal{L} = \underbrace{\|h - \hat{h}\|^2}_{\text{重建损失}} + \underbrace{\lambda \|z\|_1}_{\text{稀疏惩罚}}
$$

其中 $\hat{h} = D z$ 是重建激活，$\lambda$ 控制稀疏度。

### 4.3 特征的可解释性

SAE 学到的特征通常比神经元更具单一语义：

- 某个特征在提到"金门大桥"时激活。
- 某个特征在 Python 代码中的缩进位置激活。
- 某个特征检测到"欺骗/讽刺"语气。

通过检查激活该特征的输入样本，可以人工标注特征含义。Anthropic 在 Claude 3 Sonnet 上提取了数百万个可解释特征。

### 4.4 SAE 的扩展

- **JumpReLU SAE**：用硬阈值替代 L1 惩罚，改善稀疏性和重建质量。
- **TopK SAE**：每次只激活 Top-K 个特征，固定激活数量，训练更稳定。
- **分层 SAE**：在不同层级和残差流位置分别训练 SAE，构建完整的特征层次。
- **SAE 特征上的电路**：在稀疏特征空间而非神经元空间分析电路，可解释性更强。
- **自动可解释性**：用 LLM 自动生成特征描述并打分（Bills et al., 2023），但准确率有限。

---

## 5. 表征工程与 Steering

### 5.1 表征工程（Representation Engineering）

表征工程（Zou et al., 2023）通过操控模型内部表征来控制行为：

1. 收集不同行为状态下的激活（如"诚实回答" vs "说谎"）。
2. 提取行为方向 $\mathbf{v} = \bar{h}_{\text{honest}} - \bar{h}_{\text{deceptive}}$。
3. 在推理时将该方向加到激活上：

$$
h'_i = h_i + \alpha \mathbf{v}
$$

其中 $\alpha$ 控制引导强度。这可在不微调的情况下增强或抑制特定行为。

### 5.2 Steering Vectors

Steering vectors（Turner et al., 2023）是类似的概念，但通常从训练数据的激活差异中提取：

- 对一个输入，分别在"正常"和"添加目标后缀"的条件下前向传播，取激活差作为向量。
- 将该向量加到各层，引导模型生成方向。
- 应用：情感控制、代码质量提升、安全增强、创意引导。

### 5.3 对比激活引导（CAA）

对比激活引导（Contrastive Activation Addition）用"好-坏"对比对提取方向：

$$
\mathbf{v} = \frac{1}{N}\sum_{i=1}^{N} (h_i^+ - h_i^-)
$$

其中 $h^+$ 是期望行为的激活，$h^-$ 是不期望行为的激活。CAA 在安全、风格控制等场景有效，且比微调更轻量。

### 5.4 局限与风险

- Steering 可能产生泛化性副作用——增强诚实性可能同时改变其他行为。
- 不同层和位置的引导效果不同，需实验搜索。
- 可被对抗性引导利用——攻击者可通过添加向量绕过安全对齐。
- 因果方向尚不明确：激活方向是行为的原因还是结果。

---

## 6. 因果追踪与归因

### 6.1 因果追踪（Causal Tracing）

因果追踪（Meng et al., 2022）定位知识在模型中的存储位置：

1. 对一个事实提示（如"巴黎是___的首都"）前向传播，记录所有层激活。
2. 对输入添加噪声，记录"损坏"的激活。
3. 逐层恢复干净激活，观察哪一层恢复后模型输出正确答案。

实验表明，事实知识主要存储在中间层的 MLP 中，这为 ROME/MEMIT 模型编辑提供了定位依据（见 `model-editing.md`）。

### 6.2 归因图（Attribution Graphs）

归因方法计算模型组件（注意力头、MLP）对输出的边际贡献：

- **积分梯度（Integrated Gradients）**：沿输入到基线的路径积分梯度。
- **Shapley 值**：博弈论方法计算各组件的公平贡献。
- **Attention 归因**：结合注意力权重和梯度判断真正重要的连接。

### 6.3 函数视角（Logit Lens）

Logit Lens（nostalgebraist, 2020）将中间层激活直接用未嵌入矩阵 $W_U$ 投影到词表：

$$
\hat{p}_i = \text{softmax}(W_U h_i)
$$

这可以观察模型在每一层"正在想什么"，揭示信息逐步精炼的过程。 tuned-lens 进一步用可学习的仿射变换改进投影精度。

---

## 7. 实践建议

- **入门路径**：先在小型模型（GPT-2、Pythia）上实践 TransformerLens 库，理解 attention 头和 induction heads，再上手 SAE。
- **工具链**：TransformerLens（电路分析）、SAELens（SAE 训练与分析）、NEEL-TOOLING、CircuitVis（可视化）。
- **SAE 训练**：字典大小从 $16x$ 起步，$L_1$ 系数需搜索以平衡稀疏度（~90% 零激活）和重建质量。
- **Steering 实验**：从简单可验证的行为（情感、代码风格）开始，逐层扫描找到最佳引导层，注意控制强度避免退化。
- **安全应用**：探针和 Steering 可用于运行时监控，但不应作为唯一安全防线——它们本身可能被对抗样本绕过。
- **局限性认知**：当前可解释性方法仍处于研究阶段，对大模型的完整逆向工程尚不可行。关注结论的因果证据强度，避免过度解读相关性。

---

## 8. 参考文献

[1] C. Olah, N. Cammarata, L. Schubert, et al., *Zoom In: An Introduction to Circuits*, Distill, 2020.

[2] K. Wang, A. Variengien, A. Conmy, et al., *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small*, ICLR, 2023.

[3] C. Olsson, N. Nanda, N. Joseph, et al., *In-Context Learning and Induction Heads*, Transformer Circuits Thread, 2022.

[4] L. Bricken, T. Templeton, J. Batson, et al., *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*, Transformer Circuits Thread, 2023.

[5] N. Cunningham, H. Conmy, O. Rout, et al., *Sparse Autoencoders Find Highly Interpretable Features in Claude 3 Sonnet*, arXiv preprint, 2023.

[6] A. Zou, L. Phan, S. Chen, et al., *Representation Engineering: A Top-Down Approach to AI Transparency*, arXiv preprint, 2023.

[7] E. J. Turner, A. M. Derek, J. Hadfill, *Steering GPT-2: Activation Addition Addendum*, Alignment Forum, 2023.

[8] K. Meng, D. Bau, A. Andonian, Y. Belinkov, *Locating and Editing Factual Associations in GPT*, NeurIPS, 2022.

[9] S. Bills, N. Cammarata, D. Mossing, et al., *Language Models Can Explain Neurons in Language Models*, OpenAI, 2023.

[10] J. Schwettmann, N. Cammarata, L. Schubert, et al., *A Comprehensive Mechanistic Interpretability Explainer & Glossary*, Transformer Circuits Thread, 2024.

# AI for Science 专题

> 从蛋白质结构预测到天气预测，AI 正在重塑科学研究的范式。

---

## 目录

1. [概述](#1)
2. [结构生物学与药物发现](#2)
3. [天气与气候预测](#3)
4. [材料科学](#4)
5. [数学与定理证明](#5)
6. [物理仿真与控制](#6)
7. [实践建议](#7)
8. [参考文献](#8)

---

## 1. 概述

AI for Science（AI4S）是将深度学习应用于自然科学问题的交叉领域。近年来，AI 在多个科学领域取得了突破性进展，部分场景已达到或超越传统方法：

- **AlphaFold** 解决了蛋白质结构预测这一 50 年难题。
- **AI 天气模型**在预报速度上比传统数值方法快数个数量级。
- **大语言模型**开始辅助数学证明和科学发现。

AI 在科学中的角色包括：模拟仿真器（替代昂贵的物理模拟）、数据驱动的预测器、实验设计优化器、以及科学家的智能助手。

---

## 2. 结构生物学与药物发现

### 2.1 蛋白质结构预测

**AlphaFold2（Jumper et al., 2021）** 在 CASP14 竞赛中达到实验精度，是 AI4S 的里程碑：

- 用 Evoformer（基于注意力的架构）处理多序列比对（MSA）和成对特征。
- 结构模块（Structure Module）直接生成 3D 原子坐标。
- 通过迭代细化和等变变换确保物理合理性。

AlphaFold3 进一步扩展到配体、核酸和共价修饰，覆盖蛋白质-小分子复合物。

后续工作：

- **ESMFold（Meta, 2022）**：纯序列输入（无需 MSA），速度快 60 倍，适合宏基因组学。
- **RoseTTAFold**：三轨网络（序列、成对、3D 坐标）。
- **OmegaFold**：单序列预测，适合孤儿蛋白。

### 2.2 蛋白质设计

- **ProteinMPNN**：给定结构骨架生成氨基酸序列。
- **RFdiffusion（Baker Lab, 2023）**：用扩散模型从零设计全新蛋白质结构，可设计结合蛋白、酶和对称组装体。
- **ESM-IF**：逆折叠模型，从结构预测序列。
- **Chroma（Generate Bio）**：可编程蛋白质生成模型。

### 2.3 药物发现

- **分子对接与打分**：预测小分子与靶蛋白的结合构象和亲和力。
- **生成式分子设计**：扩散模型、VAE、自回归模型生成候选药物分子。
- **属性预测**：吸收、分布、代谢、排泄、毒性（ADMET）预测。
- **AlphaFold + 虚拟筛选**：基于结构的药物发现流程。
- 代表性公司：DeepMind Isomorphic Labs、Insilico Medicine、Recursion、Atomwise。

---

## 3. 天气与气候预测

### 3.1 AI 天气预报

传统数值天气预报（NWP）通过求解偏微分方程模拟大气，计算成本极高。AI 方法用历史气象数据训练神经网络，直接从当前观测推断未来天气：

- **Pangu-Weather（华为, 2023）**：3D Earth-specific Transformer，在多个预报指标上超越 ECMWF 传统模式，速度快 10000 倍。
- **FourCastNet（NVIDIA, 2022）**：基于 Fourier Neural Operator 的全球天气预报。
- **GraphCast（DeepMind, 2023）**：基于图神经网络的全球天气预报，在 1380 个指标中 90% 超越传统模式。
- **GenCast（DeepMind, 2024）**：扩散模型生成概率天气预报，首次在精度和校准度上全面超越集合预报。
- **Aurora（Microsoft, 2024）**：基础模型范式的大气/气候预测，可在多任务间迁移。

### 3.2 AI 气候科学

- **气候降尺度**：将粗分辨率气候模型输出细化为局部预测。
- **极端事件检测**：识别飓风、热浪、洪水等极端天气模式。
- **气候模拟加速**：用 AI 替代部分物理参数化方案。
- **碳监测**：卫星图像 + AI 追踪碳排放和碳汇。

---

## 4. 材料科学

### 4.1 晶体材料发现

- **GNoME（DeepMind, 2023）**：用 GNN 预测材料稳定性，发现 220 万种新晶体，其中 38 万种稳定，相当于人类已知稳定材料的 45 倍。
- **MatterGen（Microsoft, 2025）**：扩散模型生成具有目标属性（磁性、导电性等）的无机材料。
- **CDVAE**：变分自编码器生成晶体结构。
- **MatGL / M3GNet**：图神经网络势能函数，用于材料属性预测。

### 4.2 催化剂与电池

- **Open Catalyst Project（Meta, 2020-）**：用 GNN 预测催化反应中的吸附能和反应路径，加速可再生能源催化剂发现。
- **电池材料**：AI 预测固态电解质、电极材料性能，缩短研发周期。
- **高通量筛选**：AI + 机器人实验室（自驱动实验室）自动合成和测试候选材料。

---

## 5. 数学与定理证明

### 5.1 神经定理证明

- **AlphaGeometry（DeepMind, 2024）**：神经符号系统解决奥林匹克几何题，在 IMO 上达到银牌水平。结合符号演绎引擎和语言模型预测辅助构造。
- **AlphaProof**：基于强化学习的形式化数学证明，使用 Lean 语言。
- **LeanDojo**：开源工具包，支持在 Lean 证明助手中进行机器学习。
- **GPT-4 / Claude**：可辅助形式化证明，将自然语言数学转化为 Lean/Coq/Isabelle 代码。

### 5.2 数学发现

- **猜想生成**：AI 通过观察数学对象的模式提出新猜想（如纽结理论中的新关系）。
- **FunSearch（DeepMind, 2023）**：用 LLM 搜索数学对象的构造性程序，发现了 cap set 问题的新下界。
- **AI 辅助数学研究**：数学家开始将 LLM 作为研究助手，用于文献检索、证明探索和符号计算。

### 5.3 科学代码与仿真

- **JAX / PyTorch** 成为科学计算框架，GPU/TPU 加速物理仿真。
- **神经算子（FNO, DeepONet）**：学习偏微分方程的解算子，比传统数值方法快几个数量级。
- **AI 增强 CFD（计算流体力学）**：用神经网络加速流场模拟和湍流建模。

---

## 6. 物理仿真与控制

### 6.1 粒子物理与量子计算

- **LHC 事件筛选**：CERN 使用 CNN/Transformer 实时筛选粒子对撞事件。
- **量子计算控制**：强化学习优化量子门操作和纠错。
- **格点量子色动力学**：用流模型加速采样。

### 6.2 核聚变

- **DeepMind + EPFL（2022）**：深度强化学习在托卡马克装置中控制等离子体形状。
- **聚变模拟加速**：AI 预测等离子体不稳定性和破裂。

### 6.3 神经科学

- **脑机接口**：神经网络解码神经信号，控制假肢和光标。
- **连接组学**：CNN/Transformer 分析电镜图像中的神经连接。
- **脑活动建模**：用计算模型理解感知和认知机制。

---

## 7. 实践建议

### 7.1 入门路径

1. 从感兴趣的科学领域出发（生物/物理/化学/数学），阅读该领域的 AI4S 综述。
2. 学习等变神经网络（EGNN、SE(3)-Transformer）——这是处理 3D 科学数据的基础。
3. 动手复现一个经典项目（如 AlphaFold 推理、GraphCast 预报、分子属性预测）。
4. 关注开源社区：DeepMind、Meta FAIR、Microsoft Research 的开源模型和数据集。

### 7.2 技术栈

- **框架**：PyTorch + PyG（图网络）、JAX（高性能科学计算）。
- **领域库**：RDKit（化学）、Biopython（生物）、Astropy（天文）、Qiskit（量子）。
- **数据**：PDB（蛋白质）、Materials Project（材料）、ERA5（气象）、Qlib（金融）。
- **算力**：大模型推理需 GPU，科学训练常需多 GPU/TPU 集群。

### 7.3 研究趋势

- **基础模型范式向科学迁移**：如 Aurora、Aurora-MP 等科学基础模型，在多个科学任务间共享知识。
- **神经符号融合**：结合 AI 的模式识别和符号推理的严谨性，在数学和物理中尤其重要。
- **自驱动实验室**：AI 设计实验 → 机器人执行 → 结果回传 → AI 更新模型，形成闭环。
- **多模态科学模型**：统一处理文本、分子结构、图像、光谱、基因组等科学数据。

---

## 8. 参考文献

[1] J. Jumper, R. Evans, A. Pritzel, et al., *Highly Accurate Protein Structure Prediction with AlphaFold*, Nature, 2021.

[2] R. Abramson, J. Adler, J. Dunger, et al., *Accurate Structure Prediction of Biomolecular Interactions and Interactions with AlphaFold 3*, Nature, 2024.

[3] R. Teh, C. M. Lee, J. S. Hsu, et al., *Evolutionary-Scale Modeling of Protein Complexes with ESM3*, Science, 2024.

[4] R. Krishna, J. Ma, D. I. W. Li, et al., *General Structure-Based Protein Design with RFdiffusion*, Nature, 2024.

[5] R. Lam, A. Sanchez-Gonzalez, M. Willson, et al., *Learning Skillful Medium-Range Global Weather Prediction with GraphCast*, Science, 2023.

[6] I. V. Zhirkov, X. Shi, S. Green, et al., *GenCast: Diffusion-Based Ensemble Forecasting for Medium-Range Weather*, Nature, 2025.

[9] M. J. Buehler, V. M. Loaiza, et al., *Artificial Intelligence for Science*, Nature Reviews Materials, 2024.

[7] K. Chmiela, H. E. Sauceda, et al., *Machine Learning for Molecular Simulation*, Nature Reviews Physics, 2023.

[8] T. H. Wang, J. X. Wang, et al., *Machine Learning for Science: A Survey*, arXiv preprint, 2024.

[10] A. Davies, P. Veličković, et al., *Advancing Mathematics by Guiding Human Intuition with AI*, Nature, 2021.

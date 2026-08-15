# 深度学习学习路线图

> 从零基础到独立研究的系统化学习路径。

---

## 目录

1. [阶段一：数学基础](#1)
2. [阶段二：编程基础](#2)
3. [阶段三：机器学习入门](#3)
4. [阶段四：深度学习核心](#4)
5. [阶段五：专项深造](#5)
6. [阶段六：实战与研究](#6)
7. [持续学习](#7)

---

## 1. 阶段一：数学基础

### 线性代数 (2-4 周)
- 向量、矩阵、张量运算
- 特征值与特征向量
- 奇异值分解 (SVD)
- 主成分分析 (PCA)

**推荐资源**：
- Gilbert Strang《线性代数及其应用》
- 3Blue1Brown 线性代数视频系列

### 概率论与统计 (2-3 周)
- 概率分布、条件概率、贝叶斯定理
- 期望、方差、协方差
- 最大似然估计 (MLE)
- 最大后验估计 (MAP)

**推荐资源**：
- 《概率论与数理统计》(浙大版)
- MIT 6.041 概率系统分析

### 微积分 (2-3 周)
- 导数、偏导数、梯度
- 链式法则（反向传播的基础）
- 泰勒展开
- 最优化基础

**推荐资源**：
- 3Blue1Brown 微积分视频系列
- Khan Academy Calculus

---

## 2. 阶段二：编程基础

### Python (3-4 周)
- 基础语法、数据结构
- NumPy：矩阵运算
- Pandas：数据处理
- Matplotlib/Seaborn：数据可视化

**推荐资源**：
- [Python 官方教程](https://docs.python.org/3/tutorial/)
- 《利用 Python 进行数据分析》

### Linux 基础 (1-2 周)
- 文件系统与命令行
- 包管理 (apt, conda, pip)
- 文本处理 (grep, awk, sed)
- Shell 脚本

### 版本控制 (1 周)
- Git 基本操作
- GitHub 协作流程

---

## 3. 阶段三：机器学习入门

### 机器学习基础 (4-6 周)
- 监督学习 vs 无监督学习
- 偏差-方差权衡
- 线性回归、逻辑回归
- 决策树与随机森林
- SVM
- K-Means 聚类
- 交叉验证与模型评估

**推荐资源**：
- 吴恩达《Machine Learning》(Coursera)
- 周志华《机器学习》(西瓜书)
- 李航《统计学习方法》

### 实践项目 (2-3 周)
- Kaggle 入门比赛 (Titanic, House Prices)
- 使用 Scikit-learn 实现经典算法

---

## 4. 阶段四：深度学习核心

### 基础概念 (4-6 周)
- 神经元与激活函数
- 前馈神经网络
- 反向传播算法
- 损失函数
- 优化器 (SGD, Adam)

### 卷积神经网络 (3-4 周)
- 卷积运算与池化
- 经典架构：LeNet → AlexNet → VGG → ResNet
- 迁移学习
- 目标检测与语义分割基础

**推荐资源**：
- CS231n (Stanford)
- Dive into Deep Learning (d2l.ai)

### 循环神经网络 (2-3 周)
- RNN, LSTM, GRU
- Seq2Seq 模型
- 注意力机制

### Transformer (2-3 周)
- Self-Attention 机制
- Transformer 架构详解
- BERT, GPT 原理

**推荐资源**：
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- CS224n (Stanford)

### 深度学习框架 (同步学习)
- PyTorch（推荐首选）
- 数据处理、模型构建、训练循环

---

## 5. 阶段五：专项深造

### 选择一个方向深入：

**计算机视觉 (CV)**：
- 目标检测 (YOLO, Faster R-CNN)
- 语义分割 (U-Net, DeepLab)
- 图像生成 (GAN, Diffusion Model)
- NeRF / 3D 视觉

**自然语言处理 (NLP)**：
- 预训练语言模型
- 文本分类、序列标注
- 机器翻译、文本生成
- LLM 微调与对齐

**大模型 (LLM/MLLM)**：
- 模型架构演进
- 分布式训练
- 推理优化
- 多模态模型

**强化学习 (RL)**：
- MDP, Q-Learning
- Policy Gradient
- DQN, PPO
- RLHF

---

## 6. 阶段六：实战与研究

### 独立项目 (4-8 周)
- 选择一个有挑战性的问题
- 从数据处理到部署的完整流程
- 撰写技术博客

### 论文复现 (持续)
- 选择经典论文复现
- 理解实验设计和消融研究
- 培养批判性阅读能力

### 竞赛参与
- Kaggle 竞赛
- 天池、和鲸等国内平台

### 开源贡献
- 参与开源项目
- 提交 PR、报告 Bug

---

## 7. 持续学习

### 论文追踪
- arXiv (cs.CV, cs.CL, cs.LG)
- Papers With Code
- 顶会论文 (NeurIPS, ICML, ICLR, CVPR, ACL)

### 社区
- Twitter/X (关注研究者)
- GitHub Trending
- 知乎、Reddit (r/MachineLearning)

### 学习资源
- [Papers With Code](https://paperswithcode.com/)
- [Hugging Face](https://huggingface.co/)
- [Distill.pub](https://distill.pub/)
- [Lil'Log](https://lilianweng.github.io/)

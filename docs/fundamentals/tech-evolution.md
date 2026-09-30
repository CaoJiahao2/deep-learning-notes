# 深度学习技术演进史（2012–2025）

> 按年份梳理深度学习的技术热点、范式变迁与标志性论文/工作——从 AlexNet 到 ChatGPT，从 ResNet 到 VLA 人形机器人，理解"每一年为什么发生那些事"。

---

## 目录

1. [演进主线总览](#1)
2. [史前史（2006–2011）](#2)
3. [2012–2016：深度学习革命（视觉统治时代）](#3)
4. [2017–2019：注意力机制与预训练前夜](#4)
5. [2020–2022：生成式与对齐时代](#5)
6. [2023：开源大模型与 Agent 元年](#6)
7. [2024：多模态、推理模型与具身智能爆发](#7)
8. [2025：推理平权、Agent 落地与人形机器人](#8)
9. [双轨对照：大模型 vs 具身智能里程碑](#9)
10. [规律与启示](#10)
11. [参考文献](#11)

---

## 1 演进主线总览 {#1}

```mermaid
flowchart LR
    A["2012-2016<br/>视觉革命<br/>CNN + 监督学习"] --> B["2017-2019<br/>注意力与预训练<br/>Transformer + BERT"]
    B --> C["2020-2022<br/>生成式与对齐<br/>GPT-3 / 扩散 / RLHF"]
    C --> D["2023<br/>开源与 Agent<br/>LLaMA 生态 / AutoGPT"]
    D --> E["2024<br/>多模态与推理<br/>GPT-4o / o1 / VLA"]
    E --> F["2025<br/>推理平权与具身<br/>R1 / Deep Research / 人形"]
```

| 时代 | 核心范式 | 关键瓶颈 | 破局工作 |
|---|---|---|---|
| 2012–2016 视觉革命 | 监督学习 + CNN | 标注数据、梯度消失 | AlexNet、ResNet |
| 2017–2019 预训练前夜 | 自回归/自编码预训练 | NLP 任务碎片化 | Transformer、BERT、GPT-2 |
| 2020–2022 生成与对齐 | 大规模自监督 + RLHF | few-shot 能力、可控性 | GPT-3、DDPM、ChatGPT |
| 2023 开源与 Agent | 开源复现 + 工具调用 | 成本、生态 | LLaMA、AutoGPT、Mixtral |
| 2024–2025 推理与具身 | 测试时计算 + 具身迁移 | 数学/规划能力、物理交互 | o1/R1、π0、Helix |

**一条隐藏主线**：模态扩展的顺序——文本（2018）→ 图像（2021）→ 视频（2024）→ 动作（2025）。每一步都复用"预训练 + 后训练"的配方，同时把上一时代的瓶颈（数据、对齐、推理）变成新时代的主题。

---

## 2 史前史（2006–2011） {#2}

| 年份 | 工作 | 意义 |
|---|---|---|
| 2006 | Hinton 等，深度信念网络（DBN） | "深度学习"概念诞生，逐层预训练绕开梯度消失 |
| 2009 | ImageNet 数据集发布（Fei-Fei Li） | 为监督深度学习准备了"燃料" |
| 2010–2011 | ReLU 普及、GPU 训练 DNN（Ciresan、Hinton 组） | 语音识别 DNN-HMM 开始超越 GMM |
| 2011 | Glorot 初始化、PReLU 前身（Maxout） | 训练深度网络的基础组件就位 |

---

## 3 2012–2016：深度学习革命（视觉统治时代） {#3}

### 3.1 2012：AlexNet 时刻

**热点**：ImageNet 分类错误率从 26% 断崖跌至 15.3%。

| 工作 | 机构 | 意义 |
|---|---|---|
| **AlexNet**（Krizhevsky et al.） | Toronto | GPU + ReLU + Dropout + 数据增强，深度学习复兴的起点 |
| DNN-HMM 语音识别（Dahl/Hinton 等） | Microsoft/Toronto | 语音错误率相对下降 30%，工业界转向深度学习 |

**年度变化**：视觉领域 5 年内全面切换到 CNN；GPU 算力正式成为生产力要素。

### 3.2 2013：表示学习与深度强化学习萌芽

| 工作 | 意义 |
|---|---|
| **Word2Vec**（Mikolov et al.） | 词向量时代，"king - man + woman ≈ queen" 令业界震撼 |
| **VAE**（Kingma & Welling） | 变分推断 + 神经网络，生成模型两大流派之一 |
| **DQN**（Mnih et al., NIPS 2013） | 从像素直接玩 Atari，深度强化学习开山 |
| **R-CNN**（Girshick et al.） | CNN 进入目标检测，两阶段检测范式确立 |

### 3.3 2014：GAN 与注意力雏形

| 工作 | 意义 |
|---|---|
| **GAN**（Goodfellow et al.） | 对抗训练思想，此后十年生成模型的主线之一 |
| **Adam**（Kingma & Ba） | 成为默认优化器，"无脑能用" |
| **VGG / GoogLeNet** | 深度与多尺度两条路线，卷积架构标准化 |
| **Seq2Seq**（Sutskever et al.）+ **Bahdanau Attention** | 编码器-解码器与注意力雏形，Transformer 的直系祖先 |
| GloVe、DeepFace/FaceNet | 词表示与度量学习成熟 |

### 3.4 2015：ResNet 与基础设施年

| 工作 | 意义 |
|---|---|
| **ResNet**（He et al.） | 残差连接解决退化问题，网络深度突破 100 层，此后所有大模型都有它的影子 |
| **Batch Normalization**（Ioffe & Szegedy） | 训练稳定的标配 |
| **DQN**（Nature）、**AlphaGo 胜樊麾** | 深度强化学习进入主流视野 |
| **FCN / U-Net / YOLO** | 语义分割与实时检测框架成形 |
| **TensorFlow 开源**（Google） | 深度学习框架大战开始 |
| OpenAI 成立 | AI 实验室军备竞赛启动 |

### 3.5 2016：AlphaGo 时刻

| 工作 | 意义 |
|---|---|
| **AlphaGo 4:1 李世石** | 全球性公众事件，RL + 搜索 + 深度网络的范式验证 |
| **WaveNet**（van den Oord et al.） | 自回归音频生成，自回归范式向序列全模态扩展 |
| **TPU 公开**（Google I/O） | 专用 AI 芯片时代开启 |
| PyTorch（alpha） | 动态图框架与研究生态崛起 |

**时代小结（2012–2016）**：深度学习在视觉/语音上完成"监督学习 + 更深网络 + 更大数据"的正循环；NLP 仍以词向量 + RNN 为主，突破口正在酝酿。

---

## 4 2017–2019：注意力机制与预训练前夜 {#4}

### 4.1 2017：Transformer 元年

| 工作 | 意义 |
|---|---|
| **Attention Is All You Need**（Vaswani et al.） | 自注意力取代循环结构，可完全并行——一切大模型的地基 |
| **AlphaGo Zero** | 完全自我对弈超越所有前代，RL 极致 |
| **PPO**（Schulman et al.） | 稳定易用的策略梯度，日后 RLHF 的标准算法 |
| WGAN / CycleGAN / Progressive GAN | GAN 训练稳定性与图像翻译大爆发 |

### 4.2 2018：BERT 与 NLP 预训练时代

| 工作 | 意义 |
|---|---|
| **ELMo**（Feb） | 上下文相关词表示 |
| **GPT-1**（Jun） | 生成式预训练 + 微调范式 |
| **BERT**（Oct） | 掩码语言建模横扫 11 项 NLP 任务，"预训练-微调"成为 NLP 默认范式 |
| AlphaZero、StyleGAN、BigGAN | RL 与生成模型双线丰收 |

**年度变化**：NLP 领域从"每任务一个模型"切换到"预训练模型 + 下游微调"，与 2012 年的视觉革命同构。

### 4.3 2019：预训练军备竞赛

| 工作 | 意义 |
|---|---|
| **GPT-2**（OpenAI, Feb） | 15 亿参数，zero-shot 任务迁移初现；"太危险不全开源"引发治理讨论 |
| RoBERTa / XLNet / T5 | 预训练目标与数据规模的系统化消融 |
| **EfficientNet** | 复合缩放（深度/宽度/分辨率），"scaling 有法可依" |
| **MoCo** | 对比自监督崛起，CV 的 BERT 时刻前夜 |
| AlphaStar / Pluribus | 星际争霸与六人德州扑克 |

---

## 5 2020–2022：生成式与对齐时代 {#5}

### 5.1 2020：GPT-3 与扩散模型

| 工作 | 意义 |
|---|---|
| **Scaling Laws**（Kaplan et al., Jan） | 性能是参数/数据/算力的幂律——大模型军备竞赛的理论依据 |
| **GPT-3**（May, 175B） | in-context learning / few-shot 能力涌现，提示工程诞生 |
| **DDPM**（Ho et al., Jun）+ **DDIM**（Oct） | 扩散模型成为生成新范式，直接导向 Stable Diffusion/Sora |
| **ViT**（Dosovitskiy et al., Oct） | 图像 patch 序列化，"架构统一"开始 |
| **AlphaFold 2**（CASP14） | 蛋白质折叠基本解决，AI for Science 元年 |
| **NeRF** | 隐式神经渲染，3D 视觉新路线 |

### 5.2 2021：多模态与统一表征

| 工作 | 意义 |
|---|---|
| **CLIP / DALL-E**（OpenAI, Jan） | 图文对比学习 + 文生图，开放词汇与"语言条件化视觉"确立 |
| **LoRA**（Hu et al., Jun） | 低秩适配微调，日后开源生态微调的事实标准 |
| GitHub Copilot / **Codex** | 代码大模型产品化起点 |
| **Swin Transformer** | 层级注意力成为视觉通用骨干 |
| **MAE**（He et al., Nov） | 掩码图像建模，视觉自监督第二主线 |
| **Latent Diffusion**（Rombach et al., Dec） | 潜空间扩散——Stable Diffusion 的前身 |
| Anthropic 成立 | 安全导向实验室入场 |

### 5.3 2022：ChatGPT 与 AIGC 元年

| 工作 | 意义 |
|---|---|
| **InstructGPT**（Jan/Mar） | RLHF 三阶段（SFT→RM→PPO）让模型"听话"，对齐技术定型 |
| **Chinchilla**（DeepMind, Mar） | 修正 scaling law：等参数比下数据优先——训练配比科学化 |
| DALL-E 2 / **Imagen** / Midjourney（Apr–Jul） | 文生图质量跨过可用门槛 |
| **Stable Diffusion 开源**（Aug） | 生成能力平民化，AIGC 创业潮 |
| **Whisper**（Sep） | 开源语音识别 SOTA |
| AlphaCode / Minerva / Gato | 代码、数学、多任务通才的早期探索 |
| **RT-1**（Google, Dec） | 机器人 Transformer，13 万真机轨迹验证 scaling，具身智能新起点 |
| **ChatGPT**（Nov 30） | 5 天百万用户、2 月破亿——AI 进入公众视野的分水岭 |
| Constitutional AI（Anthropic, Dec） | RLAIF 路线提出 |

**时代小结（2020–2022）**：三条曲线同时陡升——语言（GPT-3→ChatGPT）、图像（扩散）、科学（AlphaFold）。"预训练学知识 + RLHF 学行为"成为新的默认配方。

---

## 6 2023：开源大模型与 Agent 元年 {#6}

| 工作 | 时间 | 意义 |
|---|---|---|
| **LLaMA-1 权重泄露** | Feb | 开源生态引爆点：Alpaca（3 月）、Vicuna（3 月）、ChatGLM-6B（3 月）跟进 |
| **GPT-4** | Mar 14 | 多模态、强推理，能力标杆；System Card 首次系统化安全评估 |
| **PaLM-E** | Mar | 多模态 LLM 直接做具身推理 |
| **AutoGPT / BabyAGI** | Mar/Apr | "LLM + 工具 + 循环"的 Agent 概念出圈（虽不实用） |
| **DPO**（Rafailov et al., May） | 免 RM 的对齐算法，开源社区对齐标配 |
| **QLoRA**（Dettmers et al., May） | 4-bit 微调 65B，消费级显卡微调大模型 |
| **vLLM**（Jun） | PagedAttention，开源推理服务性能跃升 |
| Function Calling（OpenAI, Jun） | 工具调用接口标准化，Agent 工程化起点 |
| **RT-2 / LLaVA** | Jul/Apr | VLA 概念定型；开源多模态指令微调基线 |
| **LLaMA 2 商用开源** | Jul 18 | 40B 级权重可商用，Meta 开源战略确立 |
| ControlNet / SDXL | Feb/Jul | 可控生成成熟，创作工作流重构 |
| **FlashAttention-2 / AWQ** | Jul | 推理与量化基础设施提速 |
| **Mistral 7B / Mixtral（MoE）** | Sep/Dec | 欧洲力量入场；MoE 开源化 |
| **RWKV / Mamba** | May/Dec | 非注意力架构回潮（线性注意力/SSM） |
| **Open X-Embodiment / RT-X** | Oct | 30+ 机构共享机器人数据，具身 ImageNet 时刻 |
| **Gemini（1.0）** | Dec 6 | Google 全模态反击开始 |

**年度变化**：从"单一大模型"到"开源生态 + Agent 框架 + 垂直应用"的三层结构；RAG、Agent、微调服务成为新职业方向。

---

## 7 2024：多模态、推理模型与具身智能爆发 {#7}

### 7.1 语言与推理

| 工作 | 时间 | 意义 |
|---|---|---|
| **Gemini 1.5** | Feb | 100 万 token 上下文商用，长上下文竞赛 |
| **Claude 3 / 3.5 Sonnet** | Mar/Jun | 首次在多项基准与 GPT-4 分庭抗礼；Artifacts 交互创新 |
| **LLaMA 3 / 3.1（405B）** | Apr/Jul | 开源追平闭源代差，8B 级成为应用默认 |
| **DeepSeek-V2** | May | MLA + MoE 把推理成本打下来，架构创新回流开源 |
| **GPT-4o** | May | 原生多模态实时交互（语音/视觉/文本一体） |
| **OpenAI o1** | Sep | **测试时计算**范式：推理链 + RL，数学/代码能力跃迁；"训练时 scaling → 推理时 scaling"的第二条曲线确立 |
| **Claude Computer Use** | Oct | 模型直接操作 GUI，Computer Use Agent 赛道开启 |
| **MCP（Model Context Protocol）** | Nov | 工具/上下文接入的开放协议，Agent 生态的"USB-C" |
| DeepSeek-V3 / o3（预告）/ Genie 2 / Veo 2 / Sora 公测 | Dec | 年末密集收官：MoE 671B 开源、推理模型代际预告、世界模型与视频生成全开 |

### 7.2 视觉生成与表征

- **视频生成大爆发**：Sora 演示（2 月）→ Kling、Luma Dream Machine、Runway Gen-3（年中）→ Sora 公测、Veo 2（12 月）；
- **架构多样化**：Mamba-2、xLSTM、Jamba（混合 SSM）、KAN——非 Transformer 探索集中发布；
- **可解释性里程碑**：Anthropic《Scaling Monosemanticity》（5 月）在生产级模型上提取数百万可解释特征（SAE）；
- **AlphaProof / AlphaGeometry**（7 月）：AI 达到 IMO 银牌水平，符号 + 神经混合推理。

### 7.3 具身智能（爆发之年）

| 工作 | 时间 | 意义 |
|---|---|---|
| Mobile ALOHA / ALOHA 2 | Jan/Apr | 低成本移动双臂遥操作，学术采集范式普及 |
| **Genie / V-JEPA** | Feb | 生成式交互环境与视频表征，世界模型两条路线 |
| NVIDIA GR00T（发布） | Mar GTC | 人形机器人基础模型计划 + Isaac 生态 |
| **OpenVLA** | Jun | 7B 开源 VLA，社区 VLA 研究的默认基座 |
| **DROID / UMI** | Jun/Apr | 开源真机数据与免机器人采集，数据基建补课 |
| Unitree G1（9.9 万元） | May | 人形硬件价格破 10 万元，高校平民化 |
| Figure 02 | Aug | 具备语音交互的人形进入工厂试点 |
| **π0** | Oct 31 | Flow Matching 动作头 + 异步执行，VLA 控制频率与平滑度跃迁 |
| **LeRobot 生态**（Hugging Face） | 全年 | 机器人数据的"transformers 时刻"：标准格式 + 开源策略库 |

---

## 8 2025：推理平权、Agent 落地与人形机器人 {#8}

### 8.1 语言与推理

| 工作 | 时间 | 意义 |
|---|---|---|
| **DeepSeek-R1** | Jan 20 | 开源 RL 推理模型达到 o1 水平，全球复现潮；"推理平权"确立 |
| o3-mini / **o3** | Jan/Apr | 测试时计算继续兑现；API 定价骤降 |
| Claude 3.7 / **Claude 4** | Feb/May | 混合推理（instant/extended thinking）成为产品形态 |
| Gemini 2.5 Pro | Mar | 百万上下文 + 原生思考 |
| Qwen3 / LLaMA 4 / QwQ-32B | Mar–Apr | 开源全面进入"混合推理 + MoE"代际 |
| **Kimi K2 / GLM-4.5 / MiniMax-M1** | Jun–Jul | 中国开源 1T 级 MoE、长上下文 Agent 能力开源化 |
| **GPT-5** | Aug | 统一推理/非推理路由的收敛尝试 |

### 8.2 Agent 产品化

- **Operator**（OpenAI, Jan）、**Deep Research**（Feb）：从"对话"到"执行"与"研究交付"；
- **Claude Code**（2 月）、**Codex CLI / 云端 Agent**（4–5 月）、Devin（3 月）：编程成为 Agent 落地最快场景，SWE-bench Verified 成为软硬通货；
- **MCP 生态爆发**：OpenAI、Google 相继宣布支持，工具生态从"各自 API"走向统一协议；
- Cursor / Cline / Aider 等把"AI 结对编程"变成开发者日常。

### 8.3 视频与世界模型

- Veo 3（5 月，原生音视频）、Kling 2.x、**Wan 2.1 开源**（2 月）：视频生成进入"可用资产"阶段；
- **Genie 3**（8 月）：可交互世界生成，世界模型与游戏引擎边界模糊；
- **V-JEPA 2**（6 月）：视频预训练表征直接服务机器人零样本规划；
- World Labs 空间智能产品化，3DGS 数字孪生走向工程。

### 8.4 具身智能（从模型到产业）

| 工作 | 时间 | 意义 |
|---|---|---|
| NVIDIA **Cosmos** | Jan | 世界基础模型 + 合成数据管线，"卖铲人"战略成型 |
| **Figure Helix** | Feb | 双系统 VLA（VLM 慢思考 + 200Hz 快控制）上真机人形，家庭级任务演示 |
| **GR00T N1** | Mar | 开源人形基础模型 + 数据蓝图（真机/视频/仿真三源） |
| **Gemini Robotics** | Mar | 前沿多模态大厂正式入场具身 |
| 智元 **GO-1** | Mar | 国产开源通用具身模型 |
| **π0.5** | Apr | 开放世界泛化（新家庭新任务），VLA 泛化边界推进 |
| Unitree G1 大规模交付 / Figure 03 | 全年 | 人形从"发布会"走向"交付量" |
| EU AI Act GPAI 义务生效 | Aug | 通用模型监管落地，合规成为工程约束 |

---

## 9 双轨对照：大模型 vs 具身智能里程碑 {#9}

| 年份 | 大模型主线 | 具身智能主线 | 共享技术 |
|---|---|---|---|
| 2012–2016 | CNN/词向量/RNN | 深度强化学习萌芽（DQN→AlphaGo） | 监督学习、GPU |
| 2017–2019 | Transformer/BERT/GPT-2 | RL 大成（AlphaGo Zero）；域随机化 sim2real | RL 算法（PPO）、自监督 |
| 2020–2021 | GPT-3/CLIP/扩散 | NeRF、Ego4D 人类视频；VLP 前夜 | ViT、大规模自监督 |
| 2022 | ChatGPT/RLHF/扩散 | **RT-1** 真机 scaling；Optimus 原型 | Transformer 全模态化 |
| 2023 | LLaMA 生态/Agent 概念/ControlNet | **RT-2**（VLM→VLA）、PaLM-E、RT-X 数据集 | 多模态 LLM、开源 |
| 2024 | GPT-4o/o1/长上下文/MCP | **OpenVLA/π0**、ALOHA/UMI/DROID、Genie | 生成式动作（扩散/FM≈扩散图像）、开源 VLM |
| 2025 | R1/Deep Research/编程 Agent | **Helix/GR00T N1/π0.5/Gemini Robotics**、人形交付 | 双系统架构、RL 后训练、世界模型 |

两条轨道的耦合方式：大模型每两年把新范式"下放"给机器人——预训练（2020→RT-1 时代）、指令对齐（2022→RT-2）、开源生态（2023→OpenVLA）、测试时计算与 RL 后训练（2024–2025→π0.5/GR00T RL）。**具身智能目前正处于"2023 年的大模型时刻"：数据基建完成、开源基座出现、产业资本涌入。**

---

## 10 规律与启示 {#10}

1. **Scaling 是元规律**：从 ResNet（2015）到 GPT-3（2020）到 o1（2024），"更多算力换能力"的幂律三次换挡（参数 → 数据 → 测试时计算），每次换挡都由 scaling law 论文预告；
2. **开源滞后闭源约 6–12 个月**，但开源决定生态走向：LLaMA（2023.2）之于应用，OpenVLA（2024.6）之于机器人，R1（2025.1）之于推理；
3. **每个范式的红利期约 3–4 年**：CNN（2012–2016）→ 预训练（2018–2022）→ 生成/对齐（2021–2024）→ 推理/具身（2024–？）；在红利后期，**基础设施（数据、评测、部署）投资回报最高**；
4. **瓶颈迁移链**：算力（2012）→ 梯度（2015）→ 标注（2018）→ 对齐（2022）→ 推理成本（2023）→ 数据墙（2024）→ 物理交互（2025）——每一代人的机会在上一代的瓶颈处；
5. **评测驱动进步**：ImageNet（2012）→ GLUE/SQuAD（2018）→ MMLU/HumanEval（2022）→ SWE-bench/GAIA（2024）→ 真机成功率（2025）；可信评测是稀缺资产；
6. **工程与科学交替主导**：算法突破（2012/2017/2020）之后必然跟 2–3 年工程消化期（框架、推理、数据管线）——工程师的红利期比想象的长。

> 学习建议：把本页当作"地图"而非"教材"。查到一个新概念时，先定位它在这条时间线上的位置（解决了哪个瓶颈、替代了什么），再回到对应专题页深入细节。

---

## 11 参考文献 {#11}

[1] A. Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS, 2012.

[2] K. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.

[3] A. Vaswani et al., *Attention Is All You Need*, NeurIPS, 2017.

[4] J. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL, 2019.

[5] T. Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS, 2020.

[6] J. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020. arXiv:2001.08361.

[7] J. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), NeurIPS, 2022.

[8] J. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS, 2020.

[9] A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML, 2021.

[10] R. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR, 2022.

[11] L. Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), NeurIPS, 2022.

[12] H. Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023. arXiv:2302.13971.

[13] R. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (DPO), NeurIPS, 2023.

[14] A. Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, CoRL, 2022.

[15] A. Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL, 2023.

[16] Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA, 2024.

[17] M. J. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL, 2024.

[18] Physical Intelligence, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.

[19] OpenAI, *Learning to Reason with LLMs* (o1), 2024. https://openai.com/index/learning-to-reason-with-llms/

[20] DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025. arXiv:2501.12948.

[21] Figure AI, *Helix: A Vision-Language-Action Model for Generalist Humanoid Robots*, 2025.

[22] NVIDIA, *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.

[23] Physical Intelligence, *π0.5: Open-World Generalist Robot Policy*, 2025.

[24] J. Jumper et al., *Highly Accurate Protein Structure Prediction with AlphaFold*, Nature, 2021.

[25] Anthropic, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*, 2024.

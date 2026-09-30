# 常用数据集速查

> 覆盖机器学习主要方向的常用数据集：图像分类、检测分割、NLP、LLM 评测、多模态、语音与音频、推荐系统、图数据、时间序列、强化学习、具身智能与 Agent 评测，并给出数据集选择的一般原则。

---

## 1. 图像分类

| 数据集 | 规模 | 类别数 | 说明 | 获取方式 |
|--------|------|--------|------|---------|
| MNIST | 70K (28×28 灰度) | 10 | 手写数字识别，入门标配 | `torchvision.datasets.MNIST` |
| Fashion-MNIST | 70K | 10 | MNIST 的替代，服装分类 | `torchvision.datasets.FashionMNIST` |
| CIFAR-10 / CIFAR-100 | 60K (32×32) | 10 / 100 | 经典小图分类基准 | `torchvision.datasets.CIFAR10/100` |
| STL-10 | 13K (96×96) | 10 | 面向自监督与小样本设定的分类 | `torchvision.datasets.STL10` |
| ImageNet-1K | 1.28M | 1000 | CV 事实标准，预训练/评测 | [image-net.org](https://www.image-net.org/)（需授权） |
| ImageNet-21K | 14M | 21841 | 大规模预训练 | [image-net.org](https://www.image-net.org/)（需授权） |
| Places365 | 1.8M | 365 | 场景识别 | [places2.csail.mit.edu](http://places2.csail.mit.edu/) |
| iNaturalist | 2.7M+ | 10K+ | 细粒度物种识别，天然长尾分布 | [inaturalist.org](https://www.inaturalist.org/) |

**选用建议**：入门阶段用 MNIST / CIFAR 验证训练流程；正式实验以 ImageNet-1K 为标准对比口径；细粒度与长尾方向选 iNaturalist。

## 2. 目标检测 / 实例分割 / 语义分割

| 数据集 | 规模 | 说明 | 获取方式 |
|--------|------|------|---------|
| PASCAL VOC (07/12) | 11K 图像 / 27K 目标 | 检测、分割经典基准 | [host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/) |
| MS COCO | 330K 图像 / 2.5M 标注 | 检测、分割、关键点、caption 全场景 | [cocodataset.org](https://cocodataset.org/) |
| Open Images V7 | 9M 图像 / 16M 边界框 | 超大规模检测与视觉关系标注 | [storage.googleapis.com/openimages](https://storage.googleapis.com/openimages/web/index.html) |
| Cityscapes | 25K 帧（5K 精标注） | 城市街景语义分割 | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |
| ADE20K | 22K 图像 / 150 类 | 通用场景解析（语义分割） | [sceneparsing.csail.mit.edu](http://sceneparsing.csail.mit.edu/) |
| LVIS | 164K 图像 / 1203 类 | 长尾实例分割基准 | [lvisdataset.org](https://www.lvisdataset.org/) |

**选用建议**：检测论文以 COCO 为主、VOC 为辅做消融；分割按场景选 Cityscapes（街景）/ ADE20K（通用）。

## 3. 自然语言处理

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| GLUE | 9 个任务 | 文本分类/推理 | 经典 NLU 基准 |
| SuperGLUE | 8 个更强任务 | NLU | BERT 时代的进阶基准 |
| SQuAD 2.0 | 150K 问答 | 机器阅读理解 | 含不可回答问题 |
| CoNLL-2003 | ~20K 句 | 命名实体识别 | NER 标准基准 |
| IMDB | 50K | 情感分析 | 影评二分类 |
| WMT 系列 | 数百万句子对 | 机器翻译 | 多语言翻译基准 |
| WikiText-103 | 1 亿+ 词 | 语言建模 | 维基百科语料，LM 评测常用 |
| AG News / 20 Newsgroups | 120K / 18K | 文本分类 | 新闻主题分类入门数据 |
| CNN / DailyMail | 300K+ 篇 | 文本摘要 | 抽取式与生成式摘要标准基准 |
| Universal Dependencies | 100+ 语言 | 句法分析 | 跨语言依存句法树库 |

## 4. LLM 评测基准

| 基准 | 能力 | 说明 |
|------|------|------|
| MMLU | 知识 / 多学科 | 57 个学科的四选一 |
| MMLU-Pro | MMLU 进阶 | 十选一，更重推理 |
| C-Eval / CMMLU | 中文知识 | 中文学科评测（52 / 67 个学科） |
| AGIEval | 人类考试 | 高考、司法考试、SAT 等标准化考试 |
| HumanEval | 代码 | 生成代码的单元测试通过率（pass@k） |
| MBPP | 代码 | 更广的基础编程任务 |
| LiveCodeBench | 代码 | 持续更新题库，抗污染 |
| GSM8K | 数学推理 | 小学级数学应用题 |
| MATH | 数学 | 竞赛级数学 |
| GPQA | 科学推理 | 研究生级「Google-proof」问答 |
| ARC (ARC-C / ARC-E) | 推理 | 常识推理选择题 |
| BBH (BIG-Bench Hard) | 综合推理 | 23 个困难任务 |
| IFEval | 指令遵循 | 可验证的指令遵循基准 |
| TruthfulQA | 事实性/幻觉 | 判断模型是否给出真实答案 |
| MT-Bench / Chatbot Arena | 对话能力 | 多轮对话评测与人类偏好 Elo 排名 |

## 5. 多模态 / 视觉语言

| 数据集 | 规模 | 说明 |
|--------|------|------|
| CLIP (WIT) | 400M 图文对 | 图文对比学习预训练 |
| LAION-5B | 5B 图文对 | 大规模图文预训练（注意版权/安全过滤） |
| SBU Captions / COCO Captions | 1M / 330K | 图像描述 |
| VQA v2.0 / GQA | 1.1M+ / 22M | 视觉问答，GQA 强调组合推理 |
| TextVQA / DocVQA | — | 场景文字 / 文档文字理解 |
| ScienceQA | 21K | 多模态科学问答（含讲解） |
| LLaVA-Instruct | 158K | 多模态指令微调数据 |
| MMMU | 11.5K | 大学级多模态多学科评测 |
| MMBench / MME | — | 综合多模态能力评测 |

## 6. 语音与音频

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| LibriSpeech | 1000 小时 | ASR | 英语有声书朗读，ASR 标准基准 |
| Common Voice | 2 万+ 小时 | ASR | Mozilla 众包，100+ 语言持续扩展 |
| AISHELL-1 | 178 小时 | ASR | 中文普通话朗读基准 |
| WenetSpeech | 1 万+ 小时 | ASR | 大规模中文语音（网络与配音场景） |
| GigaSpeech | 1 万小时 | ASR | 英语多来源（有声书/播客/YouTube） |
| VoxCeleb 1/2 | 7000+ 说话人 | 说话人识别 | 真实场景说话人验证基准 |
| LibriTTS / VCTK | 585 / 44 小时 | TTS | 语音合成与音色建模 |
| AudioSet | 2.1M 片段 / 527 类 | 音频事件 | 基于 YouTube 片段的多标签音频分类 |
| DNS Challenge | 500+ 小时 | 语音增强 | 降噪与语音分离竞赛数据 |

**选用建议**：英语 ASR 用 LibriSpeech 做可复现对比，中文用 AISHELL-1 / WenetSpeech；工业级弱监督预训练可参考 Whisper 使用的 68 万小时弱标注语料思路。

## 7. 推荐系统

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| MovieLens (100K/1M/25M) | 10 万~2500 万评分 | 协同过滤 / 序列推荐 | 明尼苏达大学经典基准，入门首选 |
| Amazon Reviews (2014/2018/2023) | 数千万评论 | 评论 + 评分预测 | 按品类细分（Books、Electronics 等） |
| Criteo (Kaggle CTR / 1TB) | 45M 展示日志 | CTR 预估 | 工业级展示广告基准 |
| Yelp Open Dataset | 数百万评论 | 推荐 / 文本 | 本地商户评论，含社交与图结构 |
| MIND | 160K 新闻 / 100 万用户 | 新闻推荐 | 微软新闻推荐基准 |
| KuaiRand / Tenrec | 数千万交互 | 短视频推荐 | 含序列反馈与多种辅助信号 |

**选用建议**：隐式反馈（点击/曝光）与显式评分（1~5 星）要区分口径；常用指标为 Recall@K、NDCG@K、HR@K；注意按时间切分 train/test，防止用未来交互预测过去。

## 8. 图数据

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| Cora / CiteSeer / PubMed | 2.7K~19K 节点 | 节点分类 | 引文网络，GCN 入门三件套（Planetoid） |
| PPI | 56K 节点 | 节点分类 | 蛋白质相互作用图，归纳式设定 |
| Reddit | 233K 节点 | 节点分类 | 大图邻居采样训练基准（GraphSAGE） |
| ogbn-arxiv | 169K 节点 / 1.16M 边 | 节点分类 | OGB 引文网络 |
| ogbn-products | 2.4M 节点 / 61M 边 | 节点分类 | 亚马逊商品共现图，大规模基准 |
| ogbl-citation2 | 2.9M 节点 | 链接预测 | 作者—论文引用关系 |
| PCQM4Mv2 (OGB-LSC) | 3.7M 分子 | 图回归 | 量子化学性质预测 |
| TUDataset | 数百个小图 | 图分类 | ENZYMES、PROTEINS 等集合 |

**选用建议**：Cora 等小图只用于验证想法，正式结果应在 OGB 规模上报告；注意 transductive 与 inductive 设定的区别。

## 9. 时间序列

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| ETT (h1/h2/m1/m2) | 4 组电力变压器数据 | 长序列预测 | Informer 提出的预测基准 |
| ECL (Electricity) | 321 客户小时级用电 | 多变量预测 | 用电负荷预测 |
| Traffic | 862 个传感器 | 多变量预测 | 加州高速公路流量 |
| Weather | 21 个气象指标 | 多变量预测 | 气象站数据 |
| Exchange / ILI | 8 / 966 维 | 多变量预测 | 小维度对照集，检验是否死记硬背 |
| M4 / M5 | 10 万序列 / 4 万商店 | 预测竞赛 | M5 为沃尔玛销售，含层次结构 |
| Monash TS Archive | 30 个数据集 | 跨领域评测 | 统一口径的时间序列档案 |
| PEMS03/04/07/08 | 数千传感器 | 时空预测 | 交通流预测（ST-GNN 基准） |

**选用建议**：归一化统计量只能来自训练集（否则造成 look-ahead 泄漏）；报告滚动预测窗口与预测长度；Exchange 等强随机序列可用来检验模型是否真正学到规律。

## 10. 强化学习

### 环境与基准

| 数据集/环境 | 说明 |
|------------|------|
| Gymnasium (OpenAI Gym) | 经典控制与算法环境套件（CartPole、LunarLander 等） |
| Atari (ALE) | 离散控制基准（Breakout、Pong 等 57 个游戏） |
| MuJoCo | 物理仿真连续控制（HalfCheetah、Walker2d 等） |
| DeepMind Control Suite | 与 MuJoCo 并列的连续控制套件，像素/状态两种观测 |
| ProcGen | 程序化生成环境，考察泛化而非记忆 |
| Meta-World | 50 个机械臂操作任务，多任务/元强化学习 |
| Isaac Lab | GPU 大规模并行物理仿真，机器人 RL |

### 离线 RL 与 LLM 相关

| 数据集/环境 | 说明 |
|------------|------|
| D4RL | MuJoCo / Adroit / Kitchen 离线轨迹，离线 RL 标准基准 |
| RL Unplugged | DeepMind 出品的离线 RL 数据集集合 |
| GSM8K / MATH 训练分割 | RLVR（可验证奖励强化学习）常用数学任务 |
| HumanEval / MBPP 训练分割 | 代码 RLVR 的可验证奖励来源 |

**选用建议**：连续控制优先 MuJoCo + DeepMind Control Suite 双报告；注意 gym 版本（Gym → Gymnasium 迁移）对复现的影响。

## 11. 具身智能 / 机器人

| 数据集 | 规模 | 说明 |
|--------|------|------|
| Open X-Embodiment (OXE) | 22 种本体 / 100 万+ 轨迹 | 跨机构跨本体操作数据联盟（RT-X 项目） |
| DROID | 76 万+ 轨迹 | 多实验室采集的大规模真机操作数据 |
| BridgeData V2 | 6 万+ 轨迹 | 低成本桌面操作，VLA 常用预训练数据 |
| AgiBot World | 100 万+ 轨迹 | 智元开源的大规模真机数据（含力觉等多模态信号） |
| RT-1 / RT-2 数据 | 13 万+ episodes | Google 机器人操作数据（需申请） |
| CALVIN | 34 个任务 | 长程语言条件操作基准（仿真） |
| LIBERO | 130 个任务 | 终身学习操作基准（仿真） |
| Language Table | 8 万+ 轨迹 | 语言条件真实机器人桌面操作 |

**选用建议**：注意仿真（CALVIN、LIBERO）与真机（OXE、DROID）的差距；数据格式常为 RLDS 或 LeRobot，跨本体训练要统一动作空间与控制频率。

## 12. Agent 评测

| 基准 | 规模 | 说明 |
|------|------|------|
| SWE-bench (Verified) | 2294 / 500 个 issue | 真实 GitHub 仓库中修复 bug 的软件工程任务 |
| WebArena | 812 个任务 | 自托管真实网站栈中的网页操作 |
| GAIA | 466 个问题 | 通用助手评测：多步推理 + 工具使用 |
| AgentBench | 8 个环境 | 综合 Agent 能力评测 |
| OSWorld | 369 个任务 | 真实操作系统 GUI 操控 |
| τ-bench | 对话 + 工具 | 客服场景下的工具调用与规则遵循（含用户模拟器） |
| ToolBench / API-Bank | 1.6 万+ API | 工具调用与 API 组合能力 |
| MLE-bench | 75 个 Kaggle 竞赛 | 机器学习工程 Agent 评测 |
| TheAgentCompany | 175 个任务 | 模拟软件公司中的长程工作任务 |

**选用建议**：Agent 评测要同时报告任务成功率、平均步数、Token/工具成本与安全性；环境尽量用 Docker 固定（保证可复现），并说明 pass@k 口径。

## 13. 数据集选择原则

### 13.1 规模 vs 质量

- 大规模「噪声」数据适合预训练（Whisper、CLIP 都证明了弱监督规模的威力），小规模「精标」数据更适合评测与指令微调；
- 质量比规模更重要的场景：偏好对齐数据、Agent 轨迹数据——少量错误示范可能显著拉低模型下限；
- 实践顺序：先用小而干净的子集跑通管线并验证数据质量，再扩大规模。

### 13.2 分布偏移（Distribution Shift）

- 训练集与测试集要来自同一分布，否则实验结论无法迁移；常见的偏移来源：采集时间、地域、设备、人群、语言风格；
- 应对方式：按时间切分而非随机切分（推荐系统、时间序列必做）；报告中说明数据来源；必要时做重要性加权或领域自适应实验；
- 在 ImageNet 上训练、在医疗影像上部署是最典型的协变量偏移案例，需用目标域数据做验证。

### 13.3 数据泄漏防范

- 严格遵循官方 train/val/test 划分，测试集只碰一次；
- 特征归一化、缺失值填充、特征选择等统计量只能在训练集上计算；
- 推荐系统/时序任务按时间切分，防止「用未来预测过去」；
- 去重要跨集合进行：训练集与测试集中高度重复的样本（图像副本、翻译平行句对）会虚高指标；
- Kaggle 竞赛中的 Public/Private LB 切分就是为了发现过拟合测试集的行为。

### 13.4 许可证（License）

| 许可证 | 特点 | 使用注意 |
|--------|------|---------|
| CC0 / MIT | 可自由商用、修改、分发 | 无明显限制 |
| CC BY 4.0 | 商用需署名 | 注明出处与许可链接 |
| CC BY-NC 4.0 | 禁止商用 | 商业产品不可用作训练或评测 |
| CC BY-SA 4.0 | 衍生作品需相同协议 | 注意「传染性」 |
| 自定义研究协议 | 仅限非商业研究 | ImageNet、Cityscapes 等需注册同意条款 |

**检查清单**：数据本身的版权（爬取数据可能侵权）、标注的版权（众包平台条款）、下游用途（训练 vs 评测 vs 产品）三者的许可都要核对。

### 13.5 污染检测（Contamination Detection）

- 数据污染指评测集（或其近似副本）混入训练语料，导致评测分数虚高；
- 常用检测方法：n-gram 重叠检测（GPT-3、GSM8K 均采用 13-gram/8-gram 匹配）、embedding 相似度检索、成员推断（membership inference）；
- 缓解手段：训练前去重清洗（MinHash、SimHash）、使用持续更新的动态基准（LiveCodeBench）、构造私有 holdout 测试集、报告去重前后对比；
- 论文中报告评测结果时应说明去重策略与污染检测流程，否则分数可信度存疑。

---

## 参考文献

- The MNIST Database of Handwritten Digits, 1998
- ImageNet: A Large-Scale Hierarchical Image Database, 2009
- The PASCAL Visual Object Classes (VOC) Challenge, 2010
- Microsoft COCO: Common Objects in Context, 2014
- The Cityscapes Dataset for Semantic Urban Scene Understanding, 2016
- ADE20K: A Holistic Scene Understanding Dataset, 2017
- GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, 2018
- SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems, 2019
- SQuAD: 100,000+ Questions for Machine Comprehension of Text, 2016
- Measuring Massive Multitask Language Understanding (MMLU), 2020
- MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark, 2024
- Evaluating Large Language Models Trained on Code (HumanEval), 2021
- Training Verifiers to Solve Math Word Problems (GSM8K), 2021
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark, 2023
- Learning Transferable Visual Models From Natural Language Supervision (CLIP/WIT), 2021
- LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models, 2022
- Visual Instruction Tuning (LLaVA-Instruct), 2023
- MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark, 2023
- LibriSpeech: An ASR Corpus Based on Public Domain Audio Books, 2015
- Common Voice: A Massively-Multilingual Speech Corpus, 2020
- WenetSpeech: A 10,000+ Hour Multi-Domain Mandarin Corpus, 2022
- AudioSet: An Ontology and Human-Labeled Dataset for Audio Events, 2017
- The MovieLens Datasets: History and Context, 2015
- Image-Based Recommendations on Styles and Substitutes (Amazon Reviews), 2015
- The Criteo Dataset for Display Advertising Challenge, 2014
- MIND: A Large-scale Dataset for News Recommendation, 2020
- Collective Classification in Network Data (Cora/CiteSeer/PubMed), 2008
- The Open Graph Benchmark: Datasets and Models for Graph Machine Learning, 2020
- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (ETT), 2021
- The M4 Competition: Results, Findings, Conclusion and Way Forward, 2018
- Monash Time Series Forecasting Archive, 2021
- OpenAI Gym, 2016
- MuJoCo: A Physics Engine for Model-Based Control, 2012
- D4RL: Datasets for Deep Data-Driven Reinforcement Learning, 2020
- Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning, 2019
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models, 2023
- DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, 2024
- BridgeData V2: A Dataset for Robot Learning at Scale, 2023
- AgiBot World: A Large-Scale Embodied Intelligence Dataset, 2025
- CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks, 2022
- LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, 2023
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues?, 2023
- WebArena: A Realistic Web Environment for Building Autonomous Agents, 2023
- GAIA: A Benchmark for General AI Assistants, 2023
- AgentBench: Evaluating LLMs as Agents, 2023
- OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments, 2024
- τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, 2024
- MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering, 2024
- A Survey on Data Contamination for Large Language Models, 2024

# 常用数据集速查

> 覆盖经典机器学习、计算机视觉、NLP、大模型评估与多模态的常用数据集。

---

## 1. 图像分类

| 数据集 | 规模 | 类别数 | 说明 | 获取方式 |
|--------|------|--------|------|---------|
| MNIST | 70K (28×28 灰度) | 10 | 手写数字识别，入门标配 | `torchvision.datasets.MNIST` |
| Fashion-MNIST | 70K | 10 | MNIST 的替代，服装分类 | `torchvision.datasets.FashionMNIST` |
| CIFAR-10 / CIFAR-100 | 60K (32×32) | 10 / 100 | 经典小图分类基准 | `torchvision.datasets.CIFAR10/100` |
| ImageNet-1K | 1.28M | 1000 | CV 事实标准，预训练/评测 | [image-net.org](https://www.image-net.org/)（需授权） |
| ImageNet-21K | 14M | 21841 | 大规模预训练 | [image-net.org](https://www.image-net.org/)（需授权） |
| Places365 | 1.8M | 365 | 场景识别 | [places2.csail.mit.edu](http://places2.csail.mit.edu/) |

## 2. 目标检测 / 实例分割 / 语义分割

| 数据集 | 规模 | 说明 | 获取方式 |
|--------|------|------|---------|
| PASCAL VOC | 11K 图像 / 27K 目标 | 检测、分割经典基准 | [host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/) |
| MS COCO | 330K 图像 / 2.5M 标注 | 检测、分割、关键点全场景 | [cocodataset.org](https://cocodataset.org/) |
| Cityscapes | 25K 帧 | 城市街景语义分割 | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |

## 3. 自然语言处理

| 数据集 | 规模 | 任务 | 说明 |
|--------|------|------|------|
| GLUE | 9 个任务 | 文本分类/推理 | 经典 NLU 基准 |
| SuperGLUE | 8 个更强任务 | NLU | BERT 的进阶基准 |
| SQuAD 2.0 | 150K 问答 | 机器阅读理解 | 含不可回答问题 |
| WMT 系列 | 数百万句子对 | 机器翻译 | 多语言翻译基准 |
| CONLL-2003 | ~20K | 命名实体识别 | NER 标准基准 |
| IMDB | 50K | 情感分析 | 影评二分类 |

## 4. LLM 评测基准

| 基准 | 能力 | 说明 |
|------|------|------|
| MMLU | 知识 / 多学科 | 57 个学科的多选题 |
| MMLU-Pro | MMLU 进阶 | 更难的推理与多选 |
| HumanEval | 代码 | 生成代码并单元测试通过率 |
| MBPP | 代码 | 更广的编程任务 |
| GSM8K | 数学推理 | 小学级数学应用题 |
| MATH | 数学 | 竞赛级数学（AIME 等） |
| GPQA | 科学推理 | 研究生级问答 |
| ARC | 推理 | 常识推理（ARC-C / ARC-E） |
| BBH (BIG-Bench Hard) | 综合推理 | 23 个困难任务 |
| IFEval | 指令遵循 | 可验证的指令遵循基准 |
| TruthfulQA | 事实性/幻觉 | 判断模型是否给出真实答案 |

## 5. 多模态 / 视觉语言

| 数据集 | 规模 | 说明 |
|--------|------|------|
| CLIP (WIT) | 400M 图文对 | 图文对比学习预训练 |
| LAION-5B | 5B 图文对 | 大规模图文预训练（注意版权/安全过滤） |
| SBU Captions / COCO Captions | 1M / 330K | 图像描述 |
| LLaVA-Instruct | 158K | 多模态指令微调数据 |
| ScienceQA | 21K | 多模态科学问答 |
| MMMU | 11.5K | 大学级多模态多学科评测 |
| TextVQA / DocVQA | — | 文档/场景文字理解 |

## 6. 强化学习 / 智能体

| 数据集/环境 | 说明 |
|------------|------|
| OpenAI Gym / Gymnasium | 经典 RL 环境套件 |
| MuJoCo | 连续控制 |
| Atari | 离散控制基准 |
| D4RL | 离线 RL 数据集 |
| VL-Bench / GSM8K-RLVR | RLVR（可验证奖励强化学习）训练与评测 |

## 7. 使用建议

1. **学术使用**：优先 ImageNet / COCO / GLUE / MMLU 等公认基准，便于横向对比。
2. **数据版权与安全**：LAION 等大规模爬取数据需做过滤与授权核验。
3. **评测口径**：LLM 评测需固定 few-shot 数量、prompt 模板与随机种子，结果才可复现。
4. **数据划分**：务必遵循官方 train/val/test 划分，避免数据泄露。

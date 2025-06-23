# 深度学习（计算机视觉）常用资源大全

本文件系统整理了计算机视觉领域中与深度学习相关的常用网站、文档、数据集、工具库及 GitHub 优秀项目，适用于学生、研究者和工程师进行深入学习与开发实践。

---

## 目录

* [一、学习与文档资源](#一学习与文档资源)
* [二、常用数据集与平台](#二常用数据集与平台)

  * [2.1 图像分类](#21-图像分类)
  * [2.2 目标检测](#22-目标检测)
  * [2.3 语义/实例分割](#23-语义实例分割)
  * [2.4 深度估计与三维感知](#24-深度估计与三维感知)
  * [2.5 多模态与跨模态](#25-多模态与跨模态)
  * [2.6 水下/遥感/医学等特定领域](#26-水下遥感医学等特定领域)
* [三、经典与前沿 GitHub 项目](#三经典与前沿-github-项目)
* [四、工具与库](#四工具与库)
* [五、会议与论文检索](#五会议与论文检索)
* [六、博客与论坛社区](#六博客与论坛社区)

---

## 一、学习与文档资源

### 📘 官方文档（权威学习入口）

* [PyTorch](https://pytorch.org/)：Facebook 开源的深度学习框架，兼具灵活性与高效性。
* [TorchVision](https://pytorch.org/vision/stable/index.html)：PyTorch 生态中的计算机视觉扩展库，提供数据集、模型、预处理方法等。
* [OpenCV](https://docs.opencv.org/)：开源计算机视觉库，提供图像处理与传统视觉算法接口。
* [MMEngine & OpenMMLab](https://openmmlab.com/)：国内主流视觉算法平台，涵盖检测、分割、姿态估计、视频理解等。
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)：多模态 transformer 模型及其文档（如 CLIP、BLIP 等）。
* [TensorFlow](https://www.tensorflow.org/)：Google 开源的深度学习框架，适合大规模部署。
* [Keras](https://keras.io/)：高层次神经网络 API，支持 TensorFlow、Theano 等后端，易于上手。
* [Fastai](https://docs.fast.ai/)：基于 PyTorch 的高层次深度学习库，

### 📚 系统课程与学习资料

* [CS231n](http://cs231n.stanford.edu/)：斯坦福大学计算机视觉课程，系统讲解 CNN 基础与视觉任务。
* [CS143: Vision Transformers](https://vision-transformers.org/)：关注当前主流 ViT 系列模型的课程。
* [Fast.ai 深度学习速成](https://course.fast.ai/)：强调实践、适合有编程基础的学习者。
* [MIT 6.S191](https://introtodeeplearning.mit.edu/)：MIT 深度学习公开课，涵盖 Transformer、强化学习等。
* [Deep Learning Book](https://www.deeplearningbook.org/)：Ian Goodfellow 编著的深度学习经典教材。

---

## 二、常用数据集与平台

为便于查找与应用，按任务类型进行分类。

### 2.1 图像分类

* [ImageNet](http://www.image-net.org/)：大规模图像分类数据集，常用于预训练与挑战赛。
* [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)：小型分类数据集，适合模型验证与初学者入门。
* [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet)：“轻量版” ImageNet，适合资源受限的训练测试。

### 2.2 目标检测

* [COCO](https://cocodataset.org/)：包含图像分类、检测、分割和关键点标注，广泛用于检测算法评估。
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)：较早的目标检测数据集，仍用于轻量算法验证。
* [Open Images](https://storage.googleapis.com/openimages/web/index.html)：Google 提供的开放检测数据集，含有千万级标注。
* [LVIS](https://www.lvisdataset.org/)：大规模视觉识别数据集，包含丰富的长尾类别标注。
* [Cityscapes](https://www.cityscapes-dataset.com/)：专注于城市街景的检测与分割任务，适用于自动驾驶场景。
* [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)：包含丰富的场景理解标注，适用于检测与分割任务。

### 2.3 语义/实例分割

* [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)：语义分割任务中常用，支持 150 类场景理解。
* [Cityscapes](https://www.cityscapes-dataset.com/)：专注于自动驾驶城市街景分割任务。
* [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)：包含全球不同国家和环境中的街景图像。

### 2.4 深度估计与三维感知

* [KITTI](http://www.cvlibs.net/datasets/kitti/)：自动驾驶相关数据集，含深度、光流、立体视觉标注。
* [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)：室内 RGB-D 数据集，用于单目深度估计与场景理解。
* [Hypersim](https://github.com/apple/ml-hypersim)：Apple 提供的室内合成数据集，含 RGB、深度、法线等。
* [ScanNet](http://www.scan-net.org/)：3D 重建与室内场景理解数据集。

### 2.5 多模态与跨模态

* [Visual Genome](https://visualgenome.org/)：视觉问答、图像标注与关系推理常用数据集。
* [Flickr30k/COCO Captions](https://cocodataset.org/#captions-2015)：图文对标注，适用于图像字幕与跨模态检索任务。
* [VQAv2](https://visualqa.org/)：图像问答数据集，支持模型学习图文结合理解能力。

### 2.6 水下 / 遥感 / 医学等特定领域

* [UIEB](https://li-chongyi.github.io/UIEB/UIEB.html)：水下图像增强基准数据集。
* [RUIE](https://li-chongyi.github.io/proj_RUIE.html)：水下图像恢复综合评测数据集。
* [DRIVE](https://drive.grand-challenge.org/)：视网膜图像分割（医学图像）数据集。
* [DeepGlobe](http://deepglobe.org/)：遥感图像的分割、分类和道路提取。
* [SeaThru](https://csms.hacettepe.edu.tr/seathru/)：真实世界水下图像数据集，含深度与颜色参考。

---

## 三、经典与前沿 GitHub 项目

### 🔍 图像分类与 Transformer 模型

* [timm](https://github.com/huggingface/pytorch-image-models)：收录各类主流分类网络与预训练权重。
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)：基于 ResNet 架构的 CNN 改进版，与 ViT 表现媲美。
* [DINOv2](https://github.com/facebookresearch/dinov2)：自监督视觉表征预训练，兼顾分类与下游迁移。

### 🧠 检测与分割模型框架

* [MMDetection](https://github.com/open-mmlab/mmdetection)：OpenMMLab 出品的检测框架，模块化设计。
* [Detectron2](https://github.com/facebookresearch/detectron2)：Facebook 开源的检测与分割框架。
* [YOLOv8](https://github.com/ultralytics/ultralytics)：YOLO 系列最新实现，支持检测、分割与追踪。

### 🌐 多模态与生成模型

* [CLIP](https://github.com/openai/CLIP)：OpenAI 提出的图文对比学习模型。
* [BLIP-2](https://github.com/salesforce/BLIP)：多模态预训练视觉语言模型。
* [Latent Diffusion](https://github.com/CompVis/latent-diffusion)：生成式模型项目，图像生成与编辑。

---

## 四、工具与库

### 🧰 通用深度学习工具

* [NumPy](https://numpy.org/)：科学计算基础库。
* [Pandas](https://pandas.pydata.org/)：结构化数据处理。
* [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)：可视化绘图库。

### 📊 可视化与实验追踪

* [TensorBoard](https://www.tensorflow.org/tensorboard)：TensorFlow 生态可视化工具，PyTorch 同样可用。
* [Weights & Biases](https://wandb.ai/)：实验管理、超参追踪与结果可视化平台。

### 🛠️ 图像处理与标注工具

* [Albumentations](https://github.com/albumentations-team/albumentations)：强大的数据增强库。
* [Labelme](https://github.com/wkentaro/labelme)：图像手动标注工具，支持多种格式。
* [Roboflow](https://roboflow.com/)：图像数据平台，支持标注、增强与导出格式转换。

---

## 五、会议与论文检索

### 📚 论文平台

* [arXiv (cs.CV)](https://arxiv.org/list/cs.CV/recent)：最新视觉相关论文。
* [Papers With Code](https://paperswithcode.com/)：结合论文与代码，追踪 SOTA。
* [Semantic Scholar](https://www.semanticscholar.org/):：AI 驱动的学术搜索引擎，提供论文摘要与引用。
* [Google Scholar](https://scholar.google.com/):：广泛使用的学术搜索引擎，支持论文检索与引用分析。

### 🏛️ 国际顶会网站

* [CVPR](https://cvpr.thecvf.com/)：IEEE 主办，计算机视觉最顶级会议。
* [ICCV](https://iccv2023.thecvf.com/)：计算机视觉双年顶会，与 CVPR 交替举办。
* [ECCV](https://eccv2024.ecva.net/): 欧洲计算机视觉会议，每两年举办一次。
* [NeurIPS](https://nips.cc/):神经信息处理系统会议，涵盖深度学习与计算机视觉。
* [ICLR](https://iclr.cc/):国际学习表征会议，专注于深度学习理论与方法。
* [AAAI](https://aaai.org/):美国人工智能协会会议，涵盖 AI 各领域。

---

## 六、博客与论坛社区

### 🌐 技术博客

* [Distill.pub](https://distill.pub/)：以可视化方式解释复杂深度学习概念。
* [Lil'Log by Lilian Weng](https://lilianweng.github.io/lil-log/)
* [Andrej Karpathy Blog](http://karpathy.github.io/)
* [Towards Data Science](https://towardsdatascience.com/)

### 💬 社区与问答平台

* [GitHub Discussions](https://github.com/):许多开源项目的讨论区，适合提问与交流。
* [Kaggle](https://www.kaggle.com/):数据科学竞赛平台，提供数据集、代码与社区讨论。
* [OpenAI Community](https://community.openai.com/):OpenAI 官方社区，讨论 AI 相关话题。
* [Hugging Face Forum](https://discuss.huggingface.co/):Hugging Face 社区，专注于 Transformers 与多模态模型。
* [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/):   机器学习与深度学习的活跃社区，讨论最新研究与技术。
* [Stack Overflow](https://stackoverflow.com/):计算机编程问答社区，适合解决具体技术问题。
* [知乎：深度学习话题](https://www.zhihu.com/topic/19550930/hot): 深度学习相关问题与讨论，适合中文用户。

---

> 若有遗漏或更优资源，欢迎补充完善。

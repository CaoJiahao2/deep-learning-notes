# 多模态大语言模型综述

## 摘要

近年来，以GPT-4V为代表的多模态大语言模型（Multimodal Large Language Model, MLLM）已成为一个新兴的研究热点 [1]。这类模型利用强大的大语言模型（Large Language Models, LLMs）作为其"大脑"，来处理和完成复杂的多模态任务 [2]。MLLM展现出的许多令人惊讶的涌现能力，例如根据图片写故事和无OCR的数学推理 [3]，这在传统的视觉语言模型中是罕见的，暗示了其作为通往通用人工智能（AGI）的一条潜在路径。

为了追赶甚至超越GPT-4V的能力，学术界和工业界投入了大量资源进行研发，以惊人的速度推动着该领域的边界。本文旨在系统性地追踪并总结MLLM的最新进展。我们首先阐述了MLLM的基本定义和相关概念，涵盖了其核心架构、训练策略与数据，以及性能评估方法。随后，我们探讨了MLLM在支持更细粒度的交互、更多样的模态、更广泛的语言以及更丰富的应用场景方面的扩展。我们还深入讨论了多模态幻觉问题，并介绍了几种关键扩展技术，包括多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助的视觉推理（LAVR）。最后，我们总结了当前MLLM面临的挑战，并指出了未来有前景的研究方向。

**关键词**：多模态大语言模型，视觉语言模型，大语言模型，指令微调，综述

## 1. 引言

大语言模型（LLMs）近年来的发展取得了巨大成功，随着模型规模的不断扩大，它们展现出包括指令遵循、上下文学习（In-Context Learning, ICL）和思维链（Chain of Thought, CoT）在内的非凡涌现能力 [4, 5]。尽管LLMs在大多数自然语言处理任务上表现出卓越的零样本或少样本推理性能，但由于它们本质上只能理解离散的文本，因此在处理视觉信息方面存在固有"盲点"。

与此同时，大视觉模型（Large Vision Models, LVMs）虽然具备强大的视觉感知能力，但在高层次的逻辑推理方面相对滞后 [6]。鉴于这种互补性，融合LLM和LVM的"多模态大语言模型"（MLLM）应运而生。MLLM通常指基于LLM并具备接收、推理和输出多模态信息能力的模型。

与传统的生成式多模态模型相比，MLLM展现出两个显著特征：
1. MLLM基于拥有数十亿甚至更多参数的LLM，这是以往模型无法比拟的。
2. MLLM采用新颖的训练范式，如多模态指令微调（visual instruction tuning），以激发模型的全部潜能，使其能遵循新颖和复杂的人类指令 [7, 8]。

得益于这两大特征，MLLM展示了许多前所未有的能力，例如根据手绘草图生成网站代码、理解表情包的深层含义以及进行无OCR的数学推理等 [1]。本文将对MLLM的架构、训练、评估、扩展技术及挑战进行全面综述。

## 2. 模型架构

一个典型的MLLM可以被抽象为三个核心模块：一个预训练的模态编码器、一个预训练的大语言模型（LLM），以及一个连接两者的模态接口。这种结构好比人类的感知与认知系统：模态编码器如同眼睛和耳朵，负责接收并预处理视觉或听觉信号；LLM则像大脑，负责理解处理后的信号并进行推理；而模态接口则扮演着对齐不同模态信息的桥梁角色。部分模型还包含一个生成器，用以输出文本以外的其他模态 [9]。

### 2.1 模态编码器

模态编码器的主要作用是将原始的非文本信息（如图像、音频）压缩成紧凑的特征表示。研究者们通常不从头训练编码器，而是采用已经与其他模态（尤其是文本）对齐的预训练模型。例如，CLIP的视觉编码器通过在海量的图文对上进行预训练，使其视觉特征与文本语义在空间上对齐，这极大地简化了后续与LLM的对齐过程 [10]。

在选择编码器时，通常会考虑分辨率、参数量和预训练语料等因素。大量实证研究表明，使用更高分辨率的输入图像能带来显著的性能提升 [11, 12]。实现高分辨率输入的方法主要分为两类：

- **直接缩放法**：即直接将高分辨率图像输入给编码器，可能需要对编码器进行微调或替换为原生支持更高分辨率的预训练模型
- **分块法**：即将高分辨率图像切割成多个小块，连同一个降采样的全局图像一起送入编码器，从而同时捕捉局部细节和全局特征

相比之下，编码器的参数规模和训练数据构成的重要性不如输入分辨率高。除了图像，其他模态也有相应的编码器，例如使用ImageBind支持图像、文本、音频、深度、热成像和惯性测量单元（IMU）等多种模态的统一编码 [13]。

### 2.2 预训练大语言模型

直接使用一个强大的预训练LLM是构建MLLM的高效且实用的方法。这些LLM通过在海量网络文本上的预训练，已经内化了丰富的世界知识，并具备强大的泛化和推理能力。目前，开源社区中被广泛使用的LLM包括LLaMA系列 [2]、Vicuna [14]、Flan-T5 [15]等。这些模型大多以英文语料为主，而像Qwen [16]这样的模型则同时支持中英双语。

研究发现，扩大LLM的参数规模同样能带来显著的性能增益。例如，简单地将LLM从7B扩展到13B，就能在多个基准测试上实现全面提升 [11]。当使用更大规模（如34B）的LLM时，模型甚至能在仅使用英文多模态数据训练的情况下，涌现出零样本的中文处理能力。

近年来，混合专家（Mixture of Experts, MoE）架构也受到了越来越多的关注。与稠密模型相比，MoE通过选择性地激活部分参数，使得在不显著增加计算成本的前提下，大幅扩展模型的总参数量成为可能 [17]。实验证明，采用MoE架构的MLLM在几乎所有基准上都优于其对应的稠密模型 [18]。

### 2.3 模态接口

模态接口是连接不同模态和LLM的关键，其核心任务是将非文本信息投射到LLM能够理解的特征空间中。目前主流的实现方式分为两类：可学习连接器和专家模型。

#### 可学习连接器

可学习连接器根据信息融合方式的不同，又可分为令牌级融合（token-level fusion）和特征级融合（feature-level fusion）。

**令牌级融合**将编码器输出的特征转换为与文本词嵌入兼容的"软令牌"，然后与文本令牌拼接在一起送入LLM。其中：

- **基于查询（query-based）的方法**：如BLIP-2 [19]中首创并被广泛继承的Q-Former，利用一组可学习的查询向量将视觉特征压缩成固定数量的表示，从而高效地提取信息 [8, 20]
- **简洁的投影方法**：使用多层感知机（MLP）作为投影层，例如LLaVA系列就采用了单层或双层的线性MLP来对齐视觉特征和词嵌入维度 [7]

**特征级融合**则是在LLM的Transformer层之间插入额外的模块，以实现视觉与文本特征的深度交互。例如：

- **Flamingo** [21]在冻结的LLM层之间插入了交叉注意力层，用视觉线索来增强语言特征
- **CogVLM** [22]则在每个Transformer层中插入一个视觉专家模块，以实现视觉和语言特征的双向融合与交互

#### 专家模型

另一条技术路线是使用专家模型（如图像描述模型）将多模态输入直接转换成自然语言描述，然后将这些描述文本送入LLM。这种方法虽然直观，但可能在转换过程中丢失大量信息，例如，将视频转换为文本描述会破坏时空关系 [23]。

## 3. 训练策略与数据

一个成熟的MLLM通常经历三个训练阶段：预训练、指令微调和对齐微调。每个阶段都有不同的目标和数据需求，并需要精心设计的数据集来支持。

### 3.1 预训练

预训练是 MLLM 训练的第一阶段，其主要目标是对齐不同模态的特征空间，并让模型学习多模态世界知识。此阶段通常使用大规模的图文对数据，例如图像描述数据。训练任务通常是自回归的，即给定一张图像，模型需要预测其对应的文本描述，并使用标准的交叉熵损失进行优化。

一种常见的做法是冻结预训练的视觉编码器和LLM，只训练模态接口，以在保留预训练知识的同时实现高效对齐。预训练数据可分为粗粒度和细粒度两类：

- **粗粒度数据**：通常来源于网络抓取，数量庞大但描述较为简短且含有噪声，如CC [24]、SBU [25]、LAION [26]和COYO-700M [27]等数据集
- **细粒度数据**：通常包含对图像更长、更准确的描述，能够实现更精细的图文对齐

近年来，许多工作开始利用强大的闭源模型（如GPT-4V）来生成高质量的细粒度数据，例如ShareGPT4V [28]、LVIS-Instruct4V [29]和ALLAVA [30]，尽管成本更高且数据量相对较小。

### 3.2 指令微调

指令微调旨在教会模型理解并遵循用户的指令，从而完成各种指定任务。通过在多样化的指令数据上进行微调，模型能够泛化到未见过的任务，显著提升零样本性能 [15]。一个多模态指令样本通常包含一个可选的指令（任务描述）、一个多模态输入（如图像和问题）和一个期望的输出（答案）。训练目标通常是标准的自回归损失，即最大化模型生成正确响应的概率。

指令数据的收集主要有三种方式：

1. **数据适配**：即将现有的高质量任务数据集（如VQA、图像描述数据集）转换为指令格式 [8]
2. **自指令（Self-instruction）**：即利用强大的LLM（如GPT-4）根据少量人工设计的种子样本来生成大量的指令数据，LLaVA [7]是该方法的典型代表
3. **数据混合**：即将多模态指令数据与纯文本的对话数据混合训练，以提升模型的对话流畅度和指令遵循能力 [31]

研究表明，指令数据的质量比数量更为重要，其中提示的多样性（prompt diversity）和任务的覆盖范围（task coverage）是影响模型性能的关键因素。

### 3.3 对齐微调与评估

#### 3.3.1 对齐微调

对齐微调主要用于使模型的输出更符合人类的偏好，例如产生更少幻觉或更安全的回答。目前主要的技术有两种：

- **基于人类反馈的强化学习（RLHF）** [32]：通过训练一个奖励模型来学习人类对不同回答的偏好，然后使用强化学习算法（如PPO）来优化LLM的策略，使其生成能获得更高奖励的回答
- **直接偏好优化（DPO）** [33]：是一种更简洁的方法，它无需显式地训练奖励模型，而是直接通过一个简单的二元分类损失函数来学习偏好，从而简化了训练流程

用于对齐微调的数据集通常包含由人类标注的偏好对，如LLaVA-RLHF [34]和RLHF-V [35]，或者由更强大的AI模型（如GPT-4V）生成的偏好数据，如VLFeedback [36]。

#### 3.3.2 性能评估

对MLLM的评估也呈现出新的特点，即需要更全面的评估体系和针对涌现能力的新评估方案。评估方法可分为闭集评估和开集评估。

**闭集评估**通常在有预定义答案选项的任务特定数据集上进行，使用准确率、CIDEr等传统指标 [8]。为了进行更全面的比较，研究者们开发了专门为MLLM设计的基准测试，如MME [37]和MMBench [38]，它们覆盖了从基础感知到高级认知的多维度能力。

**开集评估**则针对聊天机器人等开放式回答场景，其评判标准更加复杂。常用的方法包括：

- **人工评分**：由人类评估员根据多个维度对生成内容进行打分 [31]
- **GPT评分**：利用强大的GPT模型（如GPT-4或GPT-4V）来自动评估回答的质量，这种方法因其高效而被广泛采用 [7]
- **案例研究**：通过精心设计的案例对特定模型（如GPT-4V和Gemini）进行深入的定性分析 [39]

## 4. 模型能力扩展

随着研究的深入，MLLM的能力边界不断被拓宽，覆盖了更丰富的交互方式、模态和应用场景。

### 4.1 粒度支持

为了实现更精细的人机交互，模型从支持整张图像输入，发展到支持区域级（通过边界框）[40] 甚至像素级（通过点击或涂鸦）的输入 [41, 42]。相应地，模型的输出也具备了更强的定位能力，可以生成与图像特定区域或像素掩码相关联的回答。

### 4.2 模态支持

研究人员致力于让MLLM支持更多样的输入输出模态：

- **输入方面**：模型开始支持3D点云等新模态 [43]
- **输出方面**：一些模型如NEXT-GPT [9] 已经可以借助扩散模型等技术，生成图像、音频甚至视频内容，实现了"任意到任意"的模态转换

### 4.3 语言支持

尽管目前多数模型以英语为主，但为了服务更广泛的用户，多语言MLLM的开发也取得了进展。例如，VisCPM [44] 和Qwen-VL [45] 通过利用预训练的双语LLM，并结合翻译数据进行指令微调，成功地将模型能力迁移到了中文等其他语言。

### 4.4 场景/任务扩展

研究也开始面向特定场景和专业领域进行扩展：

- **设备适配**：开发适用于手机等资源受限设备的小型化模型MobileVLM [46]
- **GUI交互**：面向图形用户界面（GUI）交互的智能体 [47]
- **专业领域应用**：将MLLM的能力应用于专业领域，如文档理解 [48]和医疗领域，例如LLaVA-Med [49] 等模型通过注入医学知识，成为了能够理解医疗影像并回答专业问题的助手

## 5. 关键技术与挑战

### 5.1 多模态幻觉

多模态幻觉是指MLLM生成的文本响应与图像内容不一致的现象，是该领域一个基础且重要的问题 [50]。幻觉可分为三种类型：

- **存在性幻觉**：描述了图像中不存在的物体
- **属性幻觉**：错误描述了物体的属性，如颜色、数量
- **关系幻觉**：错误描述了物体间的空间或交互关系 [51]

为了评估幻觉程度，研究者提出了如POPE [52] 和MME [37] 等基准。减轻幻觉的方法可分为三类：

1. **预先校正**：即通过在包含负样本的专门数据集上进行微调来抑制幻觉，例如使用包含否定性指令的数据集进行训练 [53]
2. **过程中校正**：通过改进模型架构或设计新的解码策略来在生成过程中减少幻觉 [54]
3. **事后校正**：即在生成回答后再利用外部工具或修正模型对其进行检查和纠正，例如Woodpecker [50] 框架

### 5.2 多模态上下文学习与思维链

**多模态上下文学习 (M-ICL)** 是LLM上下文学习能力的自然延伸，它允许模型通过在输入中提供少量"示例"（即图像-文本对）来学习并解决新的、未见过的任务，而无需重新训练 [21]。这种少样本学习能力在各种视觉推理任务和教会模型使用外部工具方面显示出巨大潜力 [55]。

**多模态思维链 (M-CoT)** 则是将LLM的思维链推理能力扩展到多模态领域 [5, 56]。其核心思想是引导模型在给出最终答案之前，先生成一系列中间的、有逻辑的推理步骤。这极大地增强了模型在处理需要复杂推理的科学问答等任务上的能力 [56, 57]。

获取M-CoT能力可以通过在大规模带有解释的科学问答数据集（如ScienceQA [57]）上进行微调，也可以通过在提示中加入"让我们一步一步思考"之类的指令进行零样本或少样本的引导。

### 5.3 LLM辅助的视觉推理

受工具增强型LLM成功的启发，许多研究开始探索让MLLM调用外部工具或视觉基础模型来共同完成复杂的视觉推理任务 [3, 55]。在这类系统中，LLM通常扮演"大脑"的角色，负责任务分解（将复杂问题拆解为多个子任务）、工具调用和结果汇总。

根据其具体作用，LLM可被看作：

- **控制器**：一次性规划并分配任务
- **决策者**：在多轮交互中迭代地决定下一步行动
- **语义优化器**：利用其语言能力来润色和整合来自不同工具的信息

例如，VisProg [58]通过提示GPT-3生成视觉程序来组合调用不同的视觉模块，以解决复杂问题。这种范式结合了LLM强大的推理规划能力和外部模型的专业能力，展现出卓越的泛化性和解决复杂问题的能力。

### 5.4 挑战与未来方向

尽管MLLM取得了飞速发展，但仍处于初级阶段，面临诸多挑战，这些挑战也为未来的研究指明了方向：

1. **长上下文处理能力有限**：当前模型在处理包含长视频或图文交错的长文档等多模态信息时能力有限，这限制了其在更复杂场景下的应用

2. **复杂指令遵循能力不足**：与最先进的闭源模型（如GPT-4V）相比，现有开源模型在遵循高度复杂和微妙的人类指令方面仍有较大差距

3. **关键技术尚不成熟**：如M-ICL和M-CoT等技术的潜力远未被完全挖掘，其背后的机制和改进方法仍有待深入探索

4. **具体化智能体（Embodied AI）**：开发能够与物理世界进行交互的**具体化智能体**是一个热门且极具挑战性的方向，它对模型的感知、推理、规划和执行能力提出了极高要求

5. **安全与可靠性问题**：与LLM类似，MLLM也容易受到对抗性攻击，可能被诱导产生有害或错误的输出，因此提升模型的安全性是一个至关重要的研究课题 [59]

## 6. 结论

本文对多模态大语言模型（MLLM）这一前沿领域进行了全面的综述。我们系统地梳理了其核心架构、训练范式、评估方法，并详细探讨了其在能力扩展、关键技术应用方面的前沿进展。同时，我们也指出了当前研究中存在的差距和挑战，并展望了未来的发展方向。

MLLM的时代才刚刚开启，其巨大的潜力预示着人机交互和人工智能领域即将迎来的深刻变革。我们希望本综述能为相关领域的研究人员提供一个清晰的全局视野，并激发更多创新性的研究工作。

## 参考文献

[1] OpenAI. Gpt-4 technical report. arXiv:2303.08774, 2023.

[2] Touvron, H., et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023.

[3] Yang, Z., et al. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv:2303.11381, 2023.

[4] Brown, T., et al. Language models are few-shot learners. NeurIPS, 2020.

[5] Wei, J., et al. Chain of thought prompting elicits reasoning in large language models. arXiv:2201.11903, 2022.

[6] Kirillov, A., et al. Segment anything. arXiv:2304.02643, 2023.

[7] Liu, H., Li, C., Wu, Q., & Lee, Y. J. Visual instruction tuning. arXiv:2304.08485, 2023.

[8] Dai, W., et al. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv:2305.06500, 2023.

[9] Wu, S., et al. Next-gpt: Any-to-any multimodal llm. arXiv:2309.05519, 2023.

[10] Radford, A., et al. Learning transferable visual models from natural language supervision. in ICML, 2021.

[11] Liu, H., Li, C., Li, Y., & Lee, Y. J. Improved baselines with visual instruction tuning. arXiv: 2310.03744, 2023.

[12] Liu, Y., et al. An empirical study of scaling instruct-tuned large multimodal models. arXiv:2309.09958, 2023.

[13] Han, J., et al. Imagebind-llm: Multi-modality instruction tuning. arXiv:2309.03905, 2023.

[14] Chiang, W.-L., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality. 2023.

[15] Chung, H. W., et al. Scaling instruction-finetuned language models. arXiv:2210.11416, 2022.

[16] Jin, B., et al. Qwen technical report. arXiv:2309.16609, 2023.

[17] Fedus, W., Zoph, B., & Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR, 2022.

[18] Lin, B., et al. Moe-llava: Mixture of experts for large vision-language models. arXiv:2401.15947, 2024.

[19] Li, J., et al. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv:2301.12597, 2023.

[20] Chen, F., et al. X-llm: Bootstrapping advanced large language models by treating multi-modalities as foreign languages. arXiv:2305.04160, 2023.

[21] Alayrac, J.-B., et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022.

[22] Wang, W., et al. Cogvlm: Visual expert for pretrained language models. arXiv:2311.03079, 2023.

[23] Li, K., et al. Videochat: Chat-centric video understanding. arXiv:2305.06355, 2023.

[24] Sharma, P., et al. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. in ACL, 2018.

[25] Ordonez, V., Kulkarni, G., & Berg, T. Im2text: Describing images using 1 million captioned photographs. NeurIPS, 2011.

[26] Schuhmann, C., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022.

[27] Byeon, M., et al. Coyo-700m: Image-text pair dataset. 2022.

[28] Chen, L., et al. Sharegpt4v: Improving large multi-modal models with better captions. arXiv:2311.12793, 2023.

[29] Wang, J., et al. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv:2311.07574, 2023.

[30] Chen, G. H., et al. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv:2402.11684, 2024.

[31] Ye, Q., et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv:2304.14178, 2023.

[32] Ouyang, L., et al. Training language models to follow instructions with human feedback. NeurIPS, 2022.

[33] Rafailov, R., et al. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 2023.

[34] Sun, Z., et al. Aligning large multimodal models with factually augmented rlhf. arXiv:2309.14525, 2023.

[35] Yu, T., et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. arXiv:2312.00849, 2023.

[36] Li, L., et al. Silkie: Preference distillation for large visual language models. arXiv:2312.10665, 2023.

[37] Fu, C., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.

[38] Liu, Y., et al. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023.

[39] Fu, C., et al. A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv:2312.12436.

[40] Chen, K., et al. Shikra: Unleashing multimodal ilm's referential dialogue magic. arXiv:2306.15195.

[41] Yuan, Y., et al. Osprey: Pixel understanding with visual instruction tuning. arXiv:2312.10032.

[42] You, H., et al. Ferret: Refer and ground anything anywhere at any granularity. arXiv: 2310.07704, 2023.

[43] Chen, S., et al. Ll3da: Visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. arXiv:2311.18651, 2023.

[44] Hu, J., et al. Large multilingual model models pivot zero-shot multimodal learning across languages. arXiv:2308.12038, 2023.

[45] Bai, J., et al. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023.

[46] Chu, X., et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv:2312.16886, 2023.

[47] Hong, W., et al. Cogagent: A visual language model for gui agents. arXiv:2312.08914, 2023.

[48] Ye, J., et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv:2307.02499, 2023.

[49] Li, C., et al. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. arXiv:2306.00890, 2023.

[50] Yin, S., et al. Woodpecker: Hallucination correction for multimodal large language models. arXiv: 2310.16045, 2023.

[51] Zhai, B., et al. Halle-switch: Rethinking and controlling object existence hallucinations in large vision language models for detailed caption. arXiv: 2310.01779, 2023.

[52] Li, Y., et al. Evaluating object hallucination in large vision-language models. arXiv:2305.10355, 2023.

[53] Liu, F., et al. Mitigating hallucination in large multi-modal models via robust instruction tuning. in ICLR, 2024.

[54] Leng, S., et al. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. in CVPR, 2024.

[55] Lu, P., et al. Chameleon: Plug-and-play compositional reasoning with large language models. arXiv:2304.09842, 2023.

[56] Zhang, Z., et al. Multimodal chain-of-thought reasoning in language models. arXiv:2302.00923, 2023.

[57] Lu, P., et al. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 2022.

[58] Gupta, T. & Kembhavi, A. Visual programming: Compositional visual reasoning without training. in CVPR, 2023.

[59] Zhao, Y., et al. On evaluating adversarial robustness of large vision-language models. arXiv:2305.16934, 2023.

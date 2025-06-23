# 最先进 LLM 与 MLLM 及图像/视频生成模型调研（2025 年版）

> 本文档系统梳理了 2025 年最新的开源与闭源大语言模型（LLM）、多模态语言模型（MLLM），以及图像/视频生成模型，按照模型类型与开放程度进行分类，并提供权威链接。

---

## 目录

1. [文本型 LLM（Text-only LLM）](#1-文本型-llmtext-only-llm)

   * [1.1 开源模型](#11-开源模型)
   * [1.2 闭源模型](#12-闭源模型)
2. [多模态语言模型 MLLM](#2-多模态语言模型-mllm)

   * [2.1 开源模型](#21-开源模型)
   * [2.2 闭源模型](#22-闭源模型)
3. [图像/视频生成模型](#3-图像视频生成模型)

   * [3.1 开源模型](#31-开源模型)
   * [3.2 闭源模型](#32-闭源模型)
4. [总结与建议](#4-总结与建议)

---

## 1. 文本型 LLM（Text-only LLM）

### 1.1 开源模型

* **[Meta LLaMA 3](https://ai.meta.com/llama/)**：包括 8B 与 70B 两种规模，训练数据包含多语种与代码，支持 128K context。
* **[Mistral AI（Mixtral 8x7B、Mistral Small/Medium）](https://mistral.ai/news/mixtral-of-experts/)**：强大的 MoE 架构开源模型，推理效率优，开源社区活跃。
* **[Falcon 180B](https://huggingface.co/tiiuae/falcon-180B)**：由 TII 发布的大规模 Transformer，表现优异。
* **[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)**：来自清华背景团队的研究模型，强调推理能力与压缩效率。
* **[DBRX (Databricks)](https://databricks.com/blog/introducing-dbrx-a-general-purpose-open-llm)**：MoE 架构，性能优于 LLaMA 2。
* **[Qwen 系列（Qwen 1.5/2.5/3）](https://github.com/QwenLM/Qwen)**：阿里出品，支持中文、代码、多模态输入。
* **[GPT-J / GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox)**：EleutherAI 社区实现的 GPT 替代品。

### 1.2 闭源模型

* **[GPT-4 / GPT-4o (OpenAI)](https://openai.com/index/gpt-4o/)**：支持文本、图像、音频的高级模型，推理能力极强。
* **[Claude 3 (Anthropic)](https://www.anthropic.com/index/claude)**：支持 200K 上下文，逻辑、推理任务表现优秀。
* **[Gemini 1.5/2.5 (Google DeepMind)](https://deepmind.google/technologies/gemini/)**：支持长上下文和多模态能力。
* **[ERNIE 4.0/4.5 Turbo（百度）](https://wenxin.baidu.com/ernie/)**：大语言模型 + 知识增强的代表。

---

## 2. 多模态语言模型 MLLM

### 2.1 开源模型

* **[InternVL3-78B](https://github.com/OpenGVLab/InternVL)**：多模态对齐效果强，表现出色。
* **[Aria (Shanghai AI Lab)](https://github.com/aria-vision/aria-7b)**：支持图文并茂的推理。
* **[Qwen-VL / Qwen-Omni 系列](https://github.com/QwenLM/Qwen-VL)**：多模态理解与生成能力均衡，支持图像、音频、视频。
* **[LLaVA-Next / LLaVA-1.5](https://github.com/haotian-liu/LLaVA)**：与 LLaMA 相结合的多模态视觉问答模型。
* **[VILA / VIM (Xverse)](https://github.com/X-PLUG/VILA)**：全模态对齐能力强，支持 VQA、视频问答等任务。
* **[VILA-1.5](https://github.com/X-PLUG/VILA)**：通用型视觉语言基础模型。

### 2.2 闭源模型

* **[GPT-4o (OpenAI)](https://openai.com/index/gpt-4o/)**：OpenAI 全模态旗舰，支持实时语音、图像、文本混合输入。
* **[Gemini 2.5 (Google)](https://deepmind.google/technologies/gemini/)**：支持文图音视频一体化处理。
* **[Claude 3.5 Sonnet/Opus (Anthropic)](https://www.anthropic.com/index/claude)**：可用于视觉理解、表格推理等复杂任务。
* **[ERNIE X1 / 4.5 Turbo（百度）](https://wenxin.baidu.com/ernie/)**：文图音视频一体支持，嵌入飞桨生态。

---

## 3. 图像/视频生成模型

### 3.1 开源模型

* **[Stable Diffusion XL](https://stability.ai/news/stable-diffusion-xl-release)**：最流行的开源文本生成图像模型之一。
* **[SD Turbo](https://github.com/Stability-AI/stable-diffusion-turbo)**：实时推理优化，适合部署与互动。
* **[PixArt-α (2024)](https://huggingface.co/PixArt-alpha)**：基于 DiT 架构，生成质量接近 MidJourney。
* **[Sora 风格开源模型（如 Moonshot-Vid / Open-Sora）](https://github.com/AILab-CVC/Open-Sora)**：模仿 OpenAI Sora 的视频生成模型。
* **[AnimateDiff](https://github.com/guoyww/AnimateDiff)**：基于 StableDiffusion 动画生成器。
* **[VideoCrafter2](https://github.com/VideoCrafter/VideoCrafter2)**：文本生成视频的强大开源方案。
* **[ModelScope T2V](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)**：阿里达摩院视频生成模型。

### 3.2 闭源模型

* **[Sora (OpenAI)](https://openai.com/sora)**：文本到高清视频生成的强模型，闭源但展示能力惊人。
* **[Pika Labs](https://www.pika.art/)**：实时视频生成功能，支持语义控制与风格迁移。
* **[Runway Gen-2](https://runwayml.com/)**：视频编辑与生成工具，深受艺术家欢迎。
* **[Google Lumiere](https://google-research.github.io/lumiere/)**：先进的视频生成模型，基于时空一致性优化。
* **[Kling AI（字节跳动）](https://klingai.com/)**：视频生成接近 Sora 的水准，尚处于灰度内测阶段。

---

## 4. 总结与建议

* 如果你追求 **最强性能 + 最多模态能力**，可选择 GPT-4o、Gemini、Claude 3.5。
* 偏好 **可控性 + 自主训练部署**，建议使用 Qwen、InternVL、Mistral、LLaVA 系列。
* 在图像/视频生成方面，**Stable Diffusion XL + AnimateDiff + VideoCrafter2** 是最佳开源路线，闭源方向可持续关注 **Sora / Lumiere / Kling**。
* 多模态时代已经来临，建议提前构建包括图像/文本/音频/视频的 **统一推理链路**，为构建 Agent 与 Embodied AI 做准备。

---

*本报告持续更新，后续将纳入评测指标（MMMU、MMLU、MT-Bench 等）、推理速度、部署可行性、应用案例等方面。*
> 联系与协作：欢迎通过 [OpenAI API](https://platform.openai.com) 或 [Hugging Face](https://huggingface.co/models) 获取模型接口与文档。

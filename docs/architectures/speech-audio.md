# 语音与音频大模型

> 从 ASR/TTS 级联到原生端到端语音对话，语音正成为大模型的原生交互模态。

---

## 目录

1. [概述](#1)
2. [语音识别（ASR）](#2-asr)
3. [语音合成（TTS）](#3-tts)
4. [语音 Tokenizer](#4-tokenizer)
5. [语音大模型](#5)
6. [音乐与音频生成](#6)
7. [实践建议](#7)
8. [参考文献](#8)

---

## 1. 概述

语音是人类最自然的交互方式，也是 Omni 模型的关键模态。语音 AI 经历了三个阶段：

1. **级联系统**：ASR（语音转文字）→ NLP/LLM（理解生成）→ TTS（文字转语音），各模块独立训练。
2. **语音 Token 拼接**：用神经网络 Tokenizer 将语音离散化，由 LLM 统一建模文本和语音 Token。
3. **原生端到端**：语音直接输入输出，模型内部隐式完成识别、理解和合成（GPT-4o、Moshi）。

语音处理的独特挑战：

- **实时性**：对话延迟需 < 300ms 才能自然交互。
- **流式处理**：语音是流式信号，需支持边说边听边想。
- **韵律与情感**：语音包含语调、节奏、情绪等副语言信息。
- **全双工**：人类可以同时听和说，模型也需支持打断和同时说话。

---

## 2. 语音识别（ASR）

### 2.1 从 CTC 到 Whisper

- **CTC（Connectionist Temporal Classification）**：解决输入输出未对齐问题，是早期端到端 ASR 的基础。
- **RNN-Transducer（RNN-T）**：引入预测网络和联合网络，天然支持流式识别。
- **Whisper（Radford et al., 2022, OpenAI）**：在 68 万小时多语言音频上训练的编码器-解码器 Transformer，将 ASR 重新定义为序列到序列任务。Whisper 支持 99 种语言、语音识别和语音翻译，且鲁棒性强，是当前最广泛使用的开源 ASR 模型。

### 2.2 Whisper 架构

Whisper 由音频编码器和解码器组成：

1. 音频重采样为 16kHz，转换为 80 通道 log-Mel 频谱图，以 30 秒为片段。
2. 编码器（CNN + Transformer）将频谱图编码为音频特征序列。
3. 解码器自回归生成文本 Token。
4. 特殊 Token 标注语言、是否时间戳、任务类型（转写/翻译）。

Whisper 的训练目标是标准的交叉熵：

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i | y_{<i}, x)
$$

### 2.3 现代 ASR 进展

- **Large Whisper / WhisperX**：在 Whisper 基础上加入词级时间对齐和说话人分离。
- **NVIDIA Canary-1B**：编码器-解码器模型，在识别和翻译上超越 Whisper。
- **SeamlessM4T（Meta）**：多语言多模态翻译模型，支持语音到语音、语音到文本。
- **端到端对话 ASR**：Gemini、GPT-4o 等原生多模态模型直接处理音频输入，无需独立 ASR。

---

## 3. 语音合成（TTS）

### 3.1 经典方法

- **拼接合成**：从录制语音库中拼接单元，自然度有限。
- **参数合成**：Tacotron 等模型从文本生成梅尔频谱，再用声码器（Vocoder，如 WaveNet、HiFi-GAN）转为波形。
- **VITS**：端到端 TTS，结合 VAE 和 GAN，生成自然语音。

### 3.2 大模型时代 TTS

- **VALL-E（Microsoft, 2023）**：用音频 Token（从 EnCodec）做语音合成。只需 3 秒参考音频即可克隆声音，将 TTS 重新定义为语言模型的条件生成任务。
- **Bark（Suno）**：基于 Transformer 的文本到语音模型，支持多语言、语音克隆、音乐和音效。
- **Voicebox（Meta, 2023）**：基于 Flow Matching 的非自回归 TTS，支持语音编辑、去噪和风格迁移。
- **CosyVoice（阿里）**：支持零样本语音克隆、跨语言合成和指令控制（如"用悲伤的语气读"）。
- **Fish Speech / ChatTTS**：开源中文 TTS 模型，支持对话式韵律和细粒度控制。

### 3.3 语音克隆

零样本语音克隆（Zero-shot Voice Cloning）只需几秒参考音频即可复制说话人音色：

1. 参考音频通过 Tokenizer 编码为说话人特征（说话人嵌入或离散 Token）。
2. 合成时以说话人特征为条件，生成相同音色的语音。
3. 关键挑战：保持音色一致性的同时支持不同内容、情感和语言。

---

## 4. 语音 Tokenizer

### 4.1 为什么需要语音 Tokenizer

将连续音频压缩为离散 Token 有两大优势：

- 可直接复用 LLM 的自回归架构和训练框架。
- 文本和语音可共享统一词表，实现统一建模。

### 4.2 代表性 Tokenizer

| Tokenizer | 机构 | 比特率 | 特点 |
|-----------|------|--------|------|
| SoundStream | Meta | 1.5-12 kbps | 神经音频编解码，支持流式 |
| EnCodec | Meta | 1.5-24 kbps | 多带宽、支持立体声 |
| DAC（Descript） | Descript | 1.5-16 kbps | 高保真、低延迟 |
| WavTokenizer | 开源 | 0.35-1.4 kbps | 极低码率、适合 LLM |
| Mimi（Moshi） | Kyutai | 1.1 kbps | 流式、全双工优化 |

### 4.3 技术原理

神经音频 Tokenizer 通常基于 RVQ（Residual Vector Quantization）：

1. 编码器将音频波形压缩为隐表示。
2. 第一级 VQ 量化主要内容。
3. 后续量化器逐级量化残差，还原细节：

$$
z_q = \sum_{i=1}^{Q} e_i^{(k_i)}, \quad k_i = \arg\min_j \|r_{i-1} - e_i^{(j)}\|
$$

4. 解码器从量化 Token 重建音频。
5. 训练目标包括重建损失、GAN 损失和感知损失。

---

## 5. 语音大模型

### 5.1 语音 LLM

语音 LLM 将语音 Token 与文本 Token 混合，由统一 Transformer 建模：

- **SpeechGPT**：用离散语音 Token 与文本联合建模，支持语音到语音对话。
- **Qwen-Audio / Qwen2-Audio**：音频理解模型，支持语音、音乐、声音事件。
- **Vita**：开源多模态 LLM，支持文本、图像和语音输入输出。
- **MiniCPM-o**：端侧全模态模型，支持实时语音对话。

### 5.2 原生端到端语音模型

**GPT-4o（OpenAI, 2024）**：

- 音频直接输入输出，不经过 ASR/TTS 级联。
- 延迟约 200ms，接近人类对话节奏。
- 能感知和生成情感、语调、歌唱。
- 可在对话中被打断（全双工）。

**Moshi（Kyutai, 2024）**：

- 开源全双工语音对话模型。
- 使用 Mimi Tokenizer（1.1kbps，80ms 延迟）。
- 两个并行流：用户流和模型流，支持同时听说。
- 预测用户的语音轨迹，实现真正的全双工而非简单的 VAD 切换。

### 5.3 全双工对话

全双工（Full-Duplex）意味着模型能同时听和说，像人类一样：

- 不需要等用户说完才开始思考。
- 能在用户说话时给出语气词反馈（"嗯"、"哦"）。
- 能被中途打断并立即响应。
- 技术上需要联合建模用户和模型的语音流，并处理重叠语音。

### 5.4 语音 Agent

语音正成为 Agent 的自然接口：

- 语音 + 函数调用：用户通过语音下达指令，Agent 调用工具后语音回复。
- 实时翻译 Agent：语音到语音的实时多语言对话。
- 语音 + 视觉：多模态 Agent 通过语音与用户讨论所见内容（如 GPT-4o 视觉 + 语音）。

---

## 6. 音乐与音频生成

### 6.1 音乐生成

- **MusicLM（Google, 2023）**：文本到音乐，层次化建模保持长期一致性。
- **MusicGen（Meta, 2023）**：单阶段自回归 Transformer，基于 EnCodec Token，开源且质量高。
- **Suno / Udio**：商业级文本到歌曲系统，支持歌词、旋律、编曲和人声。
- **Stable Audio**：基于扩散模型的音乐和音效生成。

### 6.2 音效与环境音

- **AudioLDM / AudioLDM 2**：文本到音频的潜扩散模型。
- **Voicebox**：支持音频去噪、编辑和风格迁移。
- **Make-An-Audio**：利用 LLM 增强文本提示进行音频生成。

### 6.3 语音编辑与后处理

- **降噪/去混响**：用模型清理录音质量。
- **语音转换（Voice Conversion, VC）**：保持内容改变音色。
- **语音编辑**：修改语音中的部分词而不重新录制整句。
- **对话增强**：去除背景噪声、分离说话人、增强可懂度。

---

## 7. 实践建议

### 7.1 模型选型

- **ASR**：Whisper large-v3 是通用首选；流式场景用 Whisper Streaming 或 faster-whisper；中文场景可考虑 Paraformer。
- **TTS**：多语言零样本克隆用 CosyVoice 或 Fish Speech；中文对话用 ChatTTS；最高质量用商业 API（ElevenLabs、Azure）。
- **语音对话**：实时性要求高用 GPT-4o Realtime API 或 Moshi；成本敏感用 ASR + LLM + TTS 级联方案。
- **端侧部署**：Whisper small/base + 量化；MiniCPM-o 作为端侧全模态方案。

### 7.2 工程建议

- 语音 Agent 必须处理 VAD（语音活动检测）、打断（barge-in）和回声消除。
- 流式 ASR 延迟优化：使用分块处理、投机解码、ONNX/TensorRT 加速。
- TTS 首包延迟是关键体验指标，可用流式声码器和首包优先策略。
- 语音克隆需注意伦理和安全：加入水印和说话人验证，防止滥用。

### 7.3 与其他主题的衔接

- 原生多模态模型中的语音处理见 `llm-mllm/unified-multimodal.md`。
- 端侧部署（Whisper.cpp、GGUF 量化）见 `deployment/edge-on-device.md`。
- 语音 Agent 的安全问题（语音钓鱼、深度伪造）见 `llm-mllm/safety-alignment.md`。

---

## 8. 参考文献

[1] A. Radford, J. W. Kim, T. Xu, et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML, 2023.

[2] C. Wang, S. Chen, Y. Wu, et al., *Neural Codec Language Models Are Zero-Shot Text to Speech Synthesizers*, arXiv preprint, 2023.

[3] M. Défossez, J. Copet, G. Synnaeve, Y. Adi, *High Fidelity Neural Audio Compression*, arXiv preprint, 2022.

[4] A. Défossez, N. Usunier, L. Ott, et al., *Snake: Better Audio Tokenization with Compressed Neural Audio Codecs*, arXiv preprint, 2024.

[5] K. Kawamura, X. Ma, A. Fang, et al., *Moshi: A Speech-Text Foundation Model for Full-Duplex Spoken Dialogue*, arXiv preprint, 2024.

[6] H. Gamper, S. Narenthiran, et al., *VoxPopuli: A Large-Scale Multilingual Speech Corpus for ASR*, ACL, 2021.

[7] Y. Fu, J. Wang, Z. Ma, et al., *CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer*, arXiv preprint, 2024.

[8] Y. Ai, H. Wu, Z. Zhang, et al., *ChatTTS: A Conversational Text-to-Speech Model*, 2024.

[9] Y. Bisk, A. Holtzman, J. Thomason, et al., *Experience Grounds Language*, EMNLP, 2020.

[10] OpenAI, *GPT-4o System Card*, 2024.

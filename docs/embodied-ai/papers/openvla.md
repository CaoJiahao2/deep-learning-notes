# OpenVLA：开源 7B Vision-Language-Action 基线

> OpenVLA 把 Llama 2、DINOv2、SigLIP 和 970k 条 OXE 轨迹组合成可下载、可微调、可复现的 7B VLA，让机器人基础模型研究进入开源时代。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | OpenVLA: An Open-Source Vision-Language-Action Model (2024) |
| 机构 | Stanford / UC Berkeley / TRI / Google DeepMind / MIT / Physical Intelligence 等 |
| 关键词 | open-source VLA、Prismatic VLM、efficient fine-tuning、LoRA、quantization |
| 定位 | VLA 研究最重要的开源基线之一 |

---

## 1 背景与问题

RT-2 证明了 VLA 有效，但模型闭源且规模巨大；研究者难以验证架构、数据和微调策略。Octo 等开源模型虽然可复现，但语义骨干和微调生态不足。

OpenVLA 的目标不是追求单一演示，而是建立一个像 Llama 之于 LLM 那样的开源 VLA 生态：权重、数据加载、训练代码、微调 notebook 和部署工具齐备。


---

## 2 核心贡献

1. 发布 7B 参数开源 VLA，基于 Prismatic-7B VLM；
2. 使用 DINOv2 + SigLIP 双视觉编码器，同时保留空间细节和语义对齐；
3. 在 OXE 的 970k 真实机器人 episodes 上训练；
4. 系统研究 LoRA、量化等参数高效微调与部署方法；
5. 在多机器人、多任务基准上超过闭源 RT-2-X 55B；
6. 开源 checkpoints、代码与 fine-tuning 流程。


---

## 3 核心原理与方法详解

OpenVLA 输出离散 action token：

$$
a_t^{(d)} \sim p_\theta(a_t^{(d)}\mid I_t, \ell, a_{<t}^{(d)}).
$$

模型组件：

```text
RGB 图像
  ├→ DINOv2：空间/几何特征
  └→ SigLIP：语义/图文对齐特征
        ↓ concat + projector
     Llama 2 7B LLM
        ↓
   action token 序列
        ↓
  de-tokenize 为机器人动作
```

DINOv2 补足 CLIP/SigLIP 在细粒度空间定位上的不足，SigLIP 提供开放词汇语义。视觉特征投影到语言 embedding 空间后与指令 token 一起输入 LLM。


---

## 4 数据与训练

- 训练数据为 OXE 中 970k 条跨机器人演示；
- 处理不同机器人本体、相机和动作空间；
- 支持多任务、多机器人 out-of-the-box 评测；
- 微调时研究 LoRA、全量微调与量化；
- 可在消费级 GPU 上做参数高效微调；
- 与 Octo、RT-2-X、Diffusion Policy、from-scratch BC 进行系统比较；
- 开源 HuggingFace checkpoint 与 PyTorch 训练栈。


---

## 5 实验、结果与解读

论文报告的关键结果：

- 在 29 个任务和多个机器人 embodiment 上，OpenVLA 比 RT-2-X 55B 高 16.5% 绝对成功率；
- 参数量约为 RT-2-X 的 $1/8$，说明开源组件与数据 mixture 设计有效；
- 微调后超过 Octo 等 pretrained policy；
- 比 from-scratch Diffusion Policy 高 20.4%；
- LoRA 微调可接近或达到全量微调效果，并大幅降低显存；
- 量化服务可保持任务成功率。


---

## 6 精读要点与实践建议

- **开源基线的价值**：统一数据格式、模型权重与微调协议，使 VLA 研究可比较；
- **视觉编码器组合**：DINOv2 的空间特征与 SigLIP 的语义特征互补；
- **微调是 VLA 落地核心**：预训练模型必须支持低成本适应；
- **工程启示**：LoRA + 量化让实验室和小团队能研究 7B 策略；
- **局限启示**：离散动作 token 的精度瓶颈后来由 π0、RDT-1B、GR00T N1 等连续生成式头改进。


---

## 7 局限与后续影响

- 离散动作 token 不擅长高频连续控制；
- 7B 模型机载推理仍需硬件优化；
- OXE 数据以桌面操作为主，接触密集与双臂数据不足；
- 开源组件的原始预训练数据并不完全开放，完全复现仍有边界；
- 后续 OpenVLA-OFT 进一步优化微调速度与成功率。


---

## 相关阅读

- [RT-X 论文精读](./rt-x.md)
- [RT-2 论文精读](./rt-2.md)
- [π0 论文精读](./pi-0.md)
- [VLA 模型总览](../vla.md)

---

## 参考文献

1. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
2. Open X-Embodiment Collaboration et al. *Open X-Embodiment*, 2023.
3. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023.
4. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.


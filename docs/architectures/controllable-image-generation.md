# 可控图像生成

> 用 ControlNet、IP-Adapter、InstantID 等"附加控制器"，把文生图模型变成"听话"的工具。

---

## 目录

1. [概述](#1)
2. [控制类型分类](#2)
3. [空间控制：ControlNet / T2I-Adapter](#3)
4. [风格与身份控制：IP-Adapter](#4)
5. [主体一致性：InstantID / PhotoMaker / Leffa](#5)
6. [编辑与姿态控制](#6)
7. [联合控制与 ComfyUI 工作流](#7)
8. [评测指标](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

文生图模型（Stable Diffusion、SDXL、Flux）只能从文本提示生成图像，难以精确控制：

- **结构布局**：边缘、深度、姿态、分割；
- **风格 / 主体身份**：参考图像中的"人脸"、"画风"、"物体"；
- **局部编辑**：把图中的某一块替换 / 修改 / 重绘。

**可控图像生成**就是给文生图模型加上一层"附加控制器"，让它在保持语义创造力的同时，严格遵循用户给定的条件。本章梳理主流方法及其代表工作。

---

## 2 控制类型分类 {#2}

| 类型 | 输入 | 代表 |
|------|------|------|
| 空间结构 | Canny 边缘、深度图、姿态、分割 | ControlNet、T2I-Adapter |
| 风格迁移 | 参考图 | IP-Adapter、Style-Adapter |
| 主体身份 | 人脸 / 物体参考 | IP-Adapter-FaceID、InstantID、PhotoMaker、Leffa |
| 局部编辑 | 蒙版 + 提示 | Blended Diffusion、Inpainting |
| 姿态驱动 | 关键点序列 | ControlNet Pose、AnimateAnyone |
| 文本编辑 | 文本掩码 | Imagic、PnP、Prompt-to-Prompt |

---

## 3 空间控制：ControlNet / T2I-Adapter {#3}

### 3.1 ControlNet

**ControlNet**（Zhang et al., 2023）把一个额外的控制网络"并联"到 U-Net 的冻结权重上：

```text
            ┌──────────────┐
control →  │ ControlNet  │ ── 零卷积 ──  + ──▶ U-Net 解码
            └──────────────┘                ▲
                                          │
                                  冻结的 UNet 权重
                                          ▲
                          text/cross-attn ← 文本 prompt
```

**关键设计**：

- **零卷积**：初始权重为零的 1×1 卷积，保证训练初期不破坏原模型；
- **权重共享**：ControlNet 主干与 UNet 编码层共享权重（小模型）或独立（Deep+Shallow）；
- **训练数据**：图像对（输入图像 → 提取边缘/深度/姿态等条件图），用扩散损失训练。

**条件类型**：Canny、Depth、Pose、Segmentation、Normal、Tile、Inpaint 等。

### 3.2 Multi-ControlNet

多个 ControlNet 可叠加，分别控制不同维度（深度 + 姿态 + 边缘）。每个 ControlNet 配一个权重，调度各控制的强度。

### 3.3 T2I-Adapter

**T2I-Adapter**（Tencent, 2023）更轻量：

- 不复制 UNet，而是用一个小型网络把控制图映射到与 UNet 特征对齐的"提示"；
- 通过 cross-attention 或 feature addition 注入；
- 推理开销比 ControlNet 低。

### 3.4 对比

| 维度 | ControlNet | T2I-Adapter |
|------|-----------|-------------|
| 参数量 | 与 UNet 相当 | 小（~70M） |
| 控制精度 | 高 | 中 |
| 训练成本 | 高 | 低 |
| 多控制 | 多 ControlNet 组合 | 多 Adapter 组合 |
| 适配能力 | 任意 UNet / DiT | 适配特定架构 |

---

## 4 风格与身份控制：IP-Adapter {#4}

### 4.1 思路

**IP-Adapter**（Ye et al., 2023）解耦了 CLIP 图像编码器，让其作为"图像 prompt"注入 UNet 的 cross-attention：

- 文本与图像分别编码，但共享 cross-attention 模块；
- 图像 prompt 与文本 prompt 在 cross-attention 中相加或拼接。

### 4.2 优势

- 与文本 prompt 兼容（"参考图 + 文字描述"）；
- 支持风格迁移、概念迁移、人物一致性。

### 4.3 IP-Adapter-FaceID

为了精确控制**人脸**身份，用 **InsightFace** 替代 CLIP image encoder：

- 提取人脸 embedding（比 CLIP image 更紧凑、更鲁棒）；
- 训练时把人脸 ID 与文本 prompt 联合训练；
- 推理时只给一张参考脸即可生成该人物的不同场景图像。

代表模型：IP-Adapter-FaceID、IP-Adapter-FaceID-PlusV2。

### 4.4 Plus / Plus V2 演进

- **Plus**：增强面部细节；
- **PlusV2**：把"人脸 + 整体形象"分离，提高可控性。

---

## 5 主体一致性：InstantID / PhotoMaker / Leffa {#5}

### 5.1 InstantID

**InstantID**（Wang et al., 2024）用一个轻量的人脸检测 + ID 提取管线，把 ID embedding 注入 UNet 的 cross-attention：

- ID 提取 → 投影 → 注入 cross-attention；
- 支持任意参考脸，零样本生成新场景下的同一人物。

### 5.2 PhotoMaker

**PhotoMaker** 强调"多张参考图 → 单一主体"，把多张参考图打包为一个"虚拟身份 token"，使生成结果在多视角下都稳定。

### 5.3 Leffa（CVPR 2025）

**Leffa**（Learning Flow Fields in Attention）：

- 显式学习注意力中的"流场"，把参考图的特征点映射到生成图；
- 在一致性（特别是服装、姿势细节）上优于纯 cross-attention 注入。

### 5.4 三者对比

| 维度 | InstantID | PhotoMaker | Leffa |
|------|-----------|------------|-------|
| 输入 | 单张/多张参考图 | 多张参考图 | 单张参考图 |
| 强项 | 零样本人脸 | 跨视角一致性 | 细节控制 |
| 风格可控 | 中 | 中 | 高 |

---

## 6 编辑与姿态控制 {#6}

### 6.1 局部编辑（Inpainting）

```text
原图 + 蒙版 + 新 prompt  →  只在蒙版内生成新内容
```

代表：Stable Diffusion Inpainting、SDXL-Inpainting。

### 6.2 姿态控制

- **ControlNet Pose**：以人体关键点（OpenPose）为条件；
- **AnimateAnyone / MagicAnimate**：用姿态序列驱动单张参考图生成动画。

### 6.3 文本引导编辑

- **Imagic**：用文本微调得到目标图像的"目标 embedding"；
- **Prompt-to-Prompt (PnP)**：通过注意力图注入实现局部编辑；
- **DragDiffusion**：用鼠标拖拽像素点做"变形编辑"。

---

## 7 联合控制与 ComfyUI 工作流 {#7}

### 7.1 多控制组合

实际场景下经常需要同时控制多个维度：

```text
参考脸 (IP-Adapter FaceID)  ─┐
姿态骨架 (ControlNet Pose)  ─┼─▶  文本 prompt + 各控制权重  ─▶  UNet  ─▶ 图像
深度图 (ControlNet Depth)   ─┘
```

### 7.2 ComfyUI / A1111 工作流

ComfyUI 把每个模型/控制封装为节点，可视化组合：

- 双 ControlNet（深度 + 边缘）；
- IP-Adapter + ControlNet + LoRA 同时使用；
- 不同的 prompt 分别控制"全局"与"局部"。

### 7.3 实用配置

- 控制强度（Control Weight）通常 0.5–1.0，过强会破坏生成质量；
- 起始 / 终止步数（Start / End Step）控制"哪几个扩散步使用条件"；
- CFG 与控制权重的耦合关系需分别调。

---

## 8 评测指标 {#8}

### 8.1 客观指标

- **CLIPScore**：图像-文本语义匹配度；
- **FaceSim**（人脸 ID 余弦相似度）：人脸一致性；
- **DINO 相似度**：图像整体结构一致性；
- **Pose Error**：与目标姿态的关节距离；
- **FID**：生成质量。

### 8.2 主观指标

- **人类偏好**：人工对图像自然度、忠实度打分；
- **A/B Test**：控制强度对生成质量的影响。

### 8.3 评测陷阱

- "忠实度"未必越高越好——强控制可能损害自然度；
- 评测基准可能与实际任务分布不同。

---

## 9 参考文献 {#9}

[1] L. Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*, ICCV, 2023. arXiv:2302.05543.

[2] H. Ye et al., *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*, 2023. arXiv:2308.06721.

[3] Q. Wang et al., *InstantID: Zero-shot Identity-Preserving Generation in Seconds*, 2024. arXiv:2401.16154.

[4] Z. Li et al., *PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding*, 2024.

[5] Z. Zhou et al., *T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models*, AAAI, 2024.

[6] Leffa Team, *Learning Flow Fields in Attention for Controllable Person Image Generation (Leffa)*, CVPR, 2025.

[7] A. Hertz et al., *Prompt-to-Prompt Image Editing with Cross Attention Control*, 2022.

[8] Z. Li et al., *DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing*, CVPR, 2024.

[9] S. Mou et al., *AnimateAnyone: Consistent and Controllable Image Animation with Motion Diffusion Models*, 2024.

[10] ComfyUI, *ComfyUI: A Stable Diffusion GUI and Backend*, 2024. https://github.com/comfyanonymous/ComfyUI.
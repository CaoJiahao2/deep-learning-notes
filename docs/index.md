# 深度学习笔记

> 一座从数学基础通往大模型、智能体与具身智能的系统化知识库。
> 每一篇笔记都遵循「概述 → 核心原理 → 数学推导 → 实践建议 → 参考文献」的结构，公式全部可编译、结论尽量可复现。

!!! tip "从哪里开始？"

    - **第一次来**：先读 [深度学习基础知识全景手册](./fundamentals/panorama.md)，建立整体地图感；
    - **系统学习**：按 [学习路线图](./fundamentals/learning-roadmap.md) 的顺序推进；
    - **查找特定内容**：使用顶部搜索（支持公式与代码高亮）。

---

## 知识版图

全站共 **八大章节**，从地基到前沿依次递进，每章都有总览页作为导航入口。

<div class="grid cards" markdown>

-   :material-map-legend:{ .lg .middle } **深度学习全景手册**

    ---

    **全站总纲**。从数学基础到大模型部署、从 CV 到多模态与 VLA/世界模型的贯通式综述，附推荐学习顺序。从这里看到整片森林，再走进每一棵树。

    [:octicons-arrow-right-24: 进入手册](./fundamentals/panorama.md)

-   :material-book-open-variant:{ .lg .middle } **基础理论**

    ---

    深度学习大厦的地基：学习路线、编程与算法、计算机系统（组成原理/OS/网络/数据库/GPU）、数学三大支柱与经典机器学习。

    [:octicons-arrow-right-24: 章节总览](./fundamentals/index.md)

-   :material-layers:{ .lg .middle } **模型架构**

    ---

    架构是能力的骨架：主干网络（Transformer/CNN/GNN/MoE）、生成模型（GAN/Diffusion/Flow Matching）、视觉理解与领域应用。

    [:octicons-arrow-right-24: 章节总览](./architectures/index.md)

-   :material-school:{ .lg .middle } **训练技术**

    ---

    架构决定上限、训练决定到达的位置：训练流程、反向传播、优化器、正则化、自监督、强化学习与分布式训练。

    [:octicons-arrow-right-24: 章节总览](./training/index.md)

-   :material-brain:{ .lg .middle } **大模型专题**

    ---

    LLM 的完整知识版图：预训练、后训练与对齐、推理与能力扩展、多模态、评测治理与前沿专题（代码模型/扩散语言模型）。

    [:octicons-arrow-right-24: 章节总览](./llm-mllm/index.md)

-   :material-robot:{ .lg .middle } **智能体**

    ---

    让模型从「回答一次」到「完成一件事」：核心循环、工具与 MCP、记忆、多智能体、编程/GUI/网页 Agent、评测与安全。

    [:octicons-arrow-right-24: 章节总览](./agents/index.md)

-   :material-robot-industrial:{ .lg .middle } **具身智能**

    ---

    智能进入物理世界：MDP/POMDP 建模、VLA 模型、策略学习、数据与仿真、世界模型、评测基准与产业应用。

    [:octicons-arrow-right-24: 章节总览](./embodied-ai/index.md)

-   :material-rocket-launch:{ .lg .middle } **推理部署**

    ---

    模型落地全链路：推理实现（KV Cache/Continuous Batching）、服务化、模型压缩、端侧 AI 与 LLMOps 应用工程。

    [:octicons-arrow-right-24: 章节总览](./deployment/index.md)

-   :material-bookshelf:{ .lg .middle } **资源汇总**

    ---

    常备工具箱：工具命令速查、数据集索引、分层学习资源、大模型求职指南与 AI for Science 专题。

    [:octicons-arrow-right-24: 章节总览](./resources/index.md)

</div>

---

## 推荐学习路径

不同背景的读者可以按下面的路径组合章节，每条路径都是一条「依赖关系正确」的阅读顺序。

### :material-routes: 路径一：零基础入门（约 3-6 个月）

```text
环境配置 → Python 与数值计算 → 数学基础 → 机器学习概述
→ CNN → Transformer → 完整训练流程 → 反向传播 → 优化器 → 正则化
```

入口：[环境配置](./fundamentals/environment-setup.md) · 主线：[基础理论](./fundamentals/index.md) → [模型架构](./architectures/index.md) → [训练技术](./training/index.md)

### :material-brain: 路径二：大模型工程师（约 2-3 个月）

```text
Transformer → 预训练 → 分词与词嵌入 → 微调技术 → 后训练综述
→ 推理优化 → 推理实现详解 → 推理服务化 → LLMOps
```

入口：[Transformer](./architectures/transformer.md) · 主线：[大模型专题](./llm-mllm/index.md) → [推理部署](./deployment/index.md)

### :material-robot: 路径三：Agent 与具身智能（约 2 个月）

```text
提示工程 → Agent 基础与核心循环 → 工具使用与 MCP → 记忆机制
→ 多智能体 → Agent 评测与安全 → 多模态对齐 → VLA 模型 → 策略学习
```

入口：[提示工程](./llm-mllm/prompt-engineering.md) · 主线：[智能体](./agents/index.md) → [具身智能](./embodied-ai/index.md)

### :material-chart-line: 路径四：研究者的快速巡礼（约 2-4 周）

```text
全景手册 → 技术演进史 → LLM/MLLM 全景调研 → 推理模型 o1/R1
→ 多模态推理综述 → 世界模型 → 机制可解释性
```

入口：[全景手册](./fundamentals/panorama.md) · 主线：[技术演进史](./fundamentals/tech-evolution.md) → 各章「前沿专题」

---

## 如何使用本站

- **公式渲染**：所有数学公式使用 KaTeX 渲染，行内为 $E = mc^2$ 风格，独立公式块居中显示；无法渲染时请检查浏览器是否加载了 CDN 脚本。
- **代码复制**：所有代码块右上角有一键复制按钮；命令行均标注了执行环境（```bash```/```python```/```text```）。
- **交叉引用**：笔记间大量互链，遇到不熟悉的先行概念，跟随链接跳转即可；浏览器后退键随时可以回来。
- **离线阅读**：本站为静态站点，可以 `mkdocs build` 后本地部署，也可以在 GitHub 上直接阅读 Markdown 源文件。

## 贡献与反馈

- 发现错误或想补充内容：欢迎提 [Issue](https://github.com/CaoJiahao2/deep-learning-notes/issues) 或 PR，贡献指南见 [CONTRIBUTING](https://github.com/CaoJiahao2/deep-learning-notes/blob/main/CONTRIBUTING.md)。
- 新文档需要登记到 `mkdocs.yml` 的 `nav:` 以及对应章节的总览页（如 `docs/llm-mllm/index.md`）。
- 所有公式必须通过 `node scripts/check_math.mjs` 校验，并通过 `mkdocs build --strict` 严格构建。

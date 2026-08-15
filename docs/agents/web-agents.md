# 网页智能体与浏览 Agent

> 从 DOM grounding 到视觉驱动浏览，从 WebVoyager 到 BrowserGym —— 让 Agent 像人一样上网办事。

---

## 目录

1. [概述](#1)
2. [网页交互的任务建模](#2)
3. [网页表示与 Grounding](#3)
4. [浏览器执行环境](#4)
5. [代表系统](#5)
6. [训练与推理范式](#6)
7. [与 GUI Agent 的关系](#7)
8. [评测与开放问题](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

网页智能体（Web Agent）让 LLM 在真实网页上执行多步任务：搜索、填表、购物、预订、发帖、数据分析。它是通用数字助手最直接的落地形态之一，也是 Agent 评测中最活跃的场景。

Web Agent 的核心能力链：

```text
理解自然语言任务
→ 将任务映射为网页操作序列
→ 在网页中定位可交互元素（grounding）
→ 执行操作（点击、输入、导航）
→ 观察页面反馈并迭代
```

与代码 Agent（`coding-agents.md`）操作仓库不同，Web Agent 操作的是**异构、动态、无固定 API 的网页**，需要更强的感知与定位能力。

---

## 2 网页交互的任务建模 {#2}

### 2.1 网页操作原语

常见操作原语（action space）：

| 操作 | 说明 |
|------|------|
| navigate(url) | 打开页面 |
| click(element) | 点击元素 |
| type(text, element) | 输入文本 |
| select(option, element) | 选择下拉选项 |
| scroll(direction) | 滚动页面 |
| hover(element) | 悬停 |
| press_key(key) | 键盘按键 |
| new_tab / switch_tab | 标签页管理 |
| go_back / go_forward | 前进后退 |

### 2.2 任务形式

- **单轮指令**：一句话完成一个任务（"帮我在 XX 上订明天早上的机票"）；
- **多轮对话**：结合历史上下文逐步完成任务；
- **多任务序列**：串联多个独立任务（"先查天气，再订酒店"）；
- **信息提取**：从网页抽取结构化信息（对比价格、收集数据）。

---

## 3 网页表示与 Grounding {#3}

Grounding 指把自然语言指令中的"元素"映射到网页中的实际可交互节点——这是 Web Agent 最关键的技术难点。

### 3.1 DOM 表示

- 将网页 DOM 树序列化为文本/结构化描述，供模型理解；
- 问题：完整 DOM 可能上万节点，超出上下文窗口；
- 需做**节点修剪**：保留可交互元素（button、input、a、select），合并文本。

### 3.2 Accessibility Tree

- 使用浏览器暴露的可访问性树（Accessibility Tree），比原始 DOM 更贴近"用户可见可交互元素"；
- 是主流 Web Agent（如 Browser-use、Mind2Web）的首选表示；
- 提供角色、名称、状态等语义信息，显著提升 grounding 效果。

### 3.3 视觉表示（Screenshot）

- 直接把页面截图输入多模态模型（视觉驱动）；
- 优点：贴近真实用户视角，无需 DOM；
- 缺点：OCR/定位误差、Token 成本高、长页面需分块；
- 混合方案：DOM/accessibility 提供结构，截图提供视觉上下文。

### 3.4 Grounding 输出

模型输出目标元素，通常三种方式：

| 方式 | 说明 | 优缺点 |
|------|------|--------|
| 元素 ID | 引用 accessibility/DOM 中元素编号 | 精确，依赖树表示 |
| 边界框 | 输出元素在截图中的坐标框 | 直观，需视觉定位 |
| 语义描述 | 用自然语言描述目标元素 | 灵活，可能有歧义 |

---

## 4 浏览器执行环境 {#4}

### 4.1 浏览器自动化

Web Agent 需要可编程控制浏览器的执行环境：

- **Playwright / Puppeteer**：主流浏览器自动化库，支持点击、输入、截图；
- **Selenium**：经典 WebDriver 自动化；
- **CDP（Chrome DevTools Protocol）**：底层浏览器控制协议；
- **BrowserGym / Lightweight Web Arena**：面向 Agent 研究的统一环境。

### 4.2 环境形态

| 形态 | 说明 |
|------|------|
| 自托管站点 | 本地部署可控网站（WebArena），可复现、可判分 |
| 真实网站 | 访问真实站点（WebVoyager），真实但不可复现 |
| 模拟器 | 浏览器内模拟（BrowserGym） |
| 无头/有头 | headless 加速，headed 便于调试与验证 |

---

## 5 代表系统 {#5}

### 5.1 研究原型

| 系统 | 特点 |
|------|------|
| Mind2Web | 跨站泛化 benchmark + 数据 |
| WebArena 环境 | 自托管任务环境 |
| WebVoyager | 多模态视觉 Web Agent |
| AutoWebGLM | 清华开源，中文网页智能体 |
| BrowserGym | 统一实验框架 |
| Agent-E | 以视觉为主的多模态网页 Agent |

### 5.2 产品级

| 产品 | 能力 |
|------|------|
| OpenAI Operator | ChatGPT 内置的网页操作 Agent（2025） |
| Anthropic Computer Use | 通用计算机操作（含浏览器） |
| Google Project Mariner | 浏览器内 Agent（Chrome 扩展） |
| Manus / Genspark | 通用任务 Agent，含大量网页操作 |
| Browser-use | 开源浏览器 Agent 框架 |

---

## 6 训练与推理范式 {#6}

### 6.1 推理时方法

- **ReAct 风格**：思考 → 操作 → 观察循环（见 `fundamentals.md`）；
- **多模态推理**：结合截图与文本共同决策；
- **反思与纠错**：任务失败后分析原因重试；
- **跨步骤规划**：先用 LLM 生成任务级计划，再逐步执行。

### 6.2 训练时方法

- **Web 数据构建**：从 HTML/历史轨迹合成指令-轨迹对（如 WebDreamer、SynWeaver 等 2025 工作）；
- **SFT**：在人类演示或合成轨迹上微调；
- **RL**：在环境中用任务完成信号做强化学习（需奖励设计）；
- **基础模型**：Qwen-UI-Agent、UI-TARS 等专门训练的 GUI/Web 基础模型。

### 6.3 在线学习与记忆

- 把成功轨迹沉淀为"技能"，供未来复用；
- 用户偏好记忆：记住常用网站、账号、填写习惯；
- 跨会话上下文：结合 `memory.md` 的记忆机制。

---

## 7 与 GUI Agent 的关系 {#7}

Web Agent 与 GUI Agent（`gui-agent.md`）重叠但不等价：

| 维度 | Web Agent | GUI Agent |
|------|-----------|-----------|
| 目标 | 浏览器内的网页任务 | 整个操作系统/桌面 |
| 定位 | DOM/accessibility/screenshot | 屏幕截图 + 坐标点击 |
| 操作 | 浏览器原语 | 键盘鼠标全局操作 |
| 技术 | grounding + 浏览器自动化 | 视觉感知 + 坐标映射 |
| 成熟度 | 较高（有统一环境与基准） | 较低（成功率普遍 < 20%） |

Web Agent 是 GUI Agent 在"浏览器子空间"的简化与落地：先掌握网页任务，再逐步泛化到整个桌面。

---

## 8 评测与开放问题 {#8}

### 8.1 评测

见 `agent-evaluation.md` 第 6 节（Mind2Web、WebArena、WebVoyager、BrowserGym）。

### 8.2 开放问题

| 问题 | 说明 |
|------|------|
| 长页面处理 | 长 DOM/滚动页面超出上下文与截图范围 |
| 动态内容 | 弹窗、异步加载、验证码干扰 |
| 跨站泛化 | 训练过的站点 vs 新站点 |
| 注入与安全 | 网页内容含恶意指令（见 `agent-security.md`） |
| 真实站点漂移 | 页面改版导致策略失效 |
| 成本 | 截图 + 长 DOM 的 token 开销大 |

---

## 9 实践建议 {#9}

- **表示选型**：优先 Accessibility Tree + 节点修剪，截图作为辅助；
- **环境落地**：开发用 BrowserGym 类环境，生产用 Playwright 控制真实浏览器；
- **安全**：限制 Agent 只操作白名单站点，外部内容按不可信处理（防注入）；
- **回退机制**：元素定位失败时提供多策略（语义搜索、坐标点击、DOM 搜索）；
- **评测闭环**：用 WebArena 类任务做回归，沉淀真实失败案例；
- **成本控制**：对长页面做分块加载，缓存已访问页面状态。

---

## 10 参考文献 {#10}

[1] X. Deng, Y. Gu, B. Zheng, et al., *Mind2Web: Towards a Generalist Agent for the Web*, NeurIPS, 2023.

[2] S. Zhou, F. F. Xu, H. Zhu, et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents*, ICLR, 2024.

[3] H. He, W. Yao, K. Ma, et al., *WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models*, ACL, 2024.

[4] J. Yoran, S. J. Moon, et al., *BrowserGym: A Unified Web Agent Environment*, 2024. arXiv:2412.05456.

[5] L. Zhan, K. Chen, et al., *AutoWebGLM: Bootstrap and Reinforce a Large Language Model-based Web Agent Locally*, 2024. arXiv:2404.03648.

[6] R. Sreedhar, et al., *Agent-E: From Autonomous Web Navigation to Foundational Design Principles in Agentic Systems*, 2025. arXiv:2507.13032.

[7] OpenAI, *Introducing Operator*, 2025.

[8] Google, *Project Mariner*, 2025.

[9] Qwen Team, *Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents*, 2025.

[10] Y. Qin, et al., *WebDreamer: Generating Web Tasks and Trajectories*, 2025.

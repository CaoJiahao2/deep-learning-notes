# 编程智能体：架构与使用

> 从 agent loop 到 sandbox，从 Codex CLI 到 Claude Code —— 编程智能体（Coding Agent）的架构拆解、常用命令与工程角色。

---

## 目录

1. [概述](#1)
2. [编程智能体的核心架构](#2)
3. [Agent Loop 与工具集](#3)
4. [上下文管理与记忆](#4)
5. [沙箱与权限模型](#5)
6. [Codex CLI 使用指南](#6)
7. [Claude Code 使用指南](#7)
8. [工程角色：Harness / Loop / Evals Engineer](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

编程智能体（Coding Agent）是把 LLM 从"代码补全器"升级为"自主软件工程师"的产物：它能读取仓库、规划修改、执行命令、运行测试、迭代修复，最终交付可用的代码变更。以 GitHub Copilot Workspace、Cursor、Devin、OpenHands、Codex CLI、Claude Code 为代表。

编程智能体与传统代码助手（autocomplete / chat）的关键区别：

```text
代码补全：  前缀 → 模型 → 后缀
Chat Copilot： 用户问 → 模型答（基于当前文件）
Coding Agent： 任务 → [读取 → 规划 → 编辑 → 执行 → 观察 → 修复] × N → 完成
```

编程智能体把"理解-修改-验证"展开为一个循环，动态获取信息、执行代码并依据结果自我修正，从而完成端到端的开发任务。

---

## 2 编程智能体的核心架构 {#2}

一个编程智能体通常由以下组件构成：

| 组件 | 作用 | 代表实现 |
|------|------|----------|
| 大脑 (LLM) | 推理、规划、生成代码 | Claude、GPT/o 系列、Qwen |
| Agent Loop | 推理-行动-观察的迭代循环 | ReAct、Plan-and-Execute |
| 工具集 (Tools) | 读写文件、执行命令、搜索 | Bash、Edit、Grep、LSP |
| 上下文 (Context) | 仓库信息、对话历史、指令 | 系统提示、CLAUDE.md/AGENTS.md |
| 沙箱 (Sandbox) | 限制命令的执行边界 | read-only / workspace-write / full |
| 编排层 (Harness) | 串起循环、会话、钩子、权限 | Codex harness、Claude Code runtime |

架构示意：

```text
用户任务
   │
   ▼
┌──────────────────────────────────────────┐
│  Harness（编排层）                         │
│  ┌────────────────────────────────────┐  │
│  │  Agent Loop                       │  │
│  │  推理 → 工具调用 → 观察 → 再推理      │  │
│  └────────────────────────────────────┘  │
│  Session / Context / Memory / Sandbox    │
└──────────────────────────────────────────┘
   │              │              │
   ▼              ▼              ▼
 文件工具         命令执行        Web/搜索/LSP
```

---

## 3 Agent Loop 与工具集 {#3}

### 3.1 Agent Loop

核心循环可形式化为：

$$\text{state}_{t+1} = f_{\text{LLM}}\left(\text{context}, \text{observation}_t\right), \quad \text{action}_t \sim \pi_{\text{LLM}}(\text{state}_t)$$

即：LLM 依据当前上下文与上次工具执行结果决定下一个动作，循环直至满足完成条件或达到轮次上限。

### 3.2 关键工具

| 工具 | 功能 | 典型命令 |
|------|------|---------|
| 文件读写 | 读取/创建/编辑文件 | `Read`, `Edit`, `Write` |
| 命令执行 | 运行构建、测试、脚本 | `Bash`, `exec` |
| 代码搜索 | 定位符号与用法 | `Grep`, `rg`, LSP 跳转 |
| 版本控制 | 查看 diff、提交 | `git diff`, `git status` |
| 结构化输出 | 返回 diff/patch | `git apply`, patch 工具 |
| 网页/文档 | 查询依赖与 API | Web search、文档抓取 |

### 3.3 编辑范式

现代编程智能体从"整文件重写"演进为"精准编辑"：

- 输出 unified diff / patch 而非完整文件，减少冲突与审查负担；
- 支持多处小编辑，保持格式与风格；
- 提供行号定位与精确替换，降低 LLM 幻觉式重写风险。

---

## 4 上下文管理与记忆 {#4}

### 4.1 项目级指令

仓库根目录的 `AGENTS.md` / `CLAUDE.md` / `README.md` 会被自动注入系统提示，作为项目的"宪法"：

```markdown
# AGENTS.md
- 使用中文写正文，专业术语保留英文
- 公式用 $$...$$ 块级、$...$ 行内
- 新增文档必须登记到 mkdocs.yml 的 nav:
```

### 4.2 会话上下文

- **对话历史**：保存在本地 session 文件，支持跨天恢复。
- **自动压缩 (Auto-compact)**：上下文接近窗口上限时自动摘要早期内容。
- **工作区范围**：限制工具可访问的目录（`--add-dir`、信任边界）。
- **MCP 工具**：通过 Model Context Protocol 接入外部工具（数据库、浏览器、内部服务）。

### 4.3 记忆

- 短期：当前会话内的文件状态与对话。
- 长期：CLAUDE.md / AGENTS.md 中的持久指令、自定义技能（skills）、插件（plugins）。

---

## 5 沙箱与权限模型 {#5}

沙箱（Sandbox）决定模型生成的命令可以在多大范围内执行，是安全性的核心机制。

### 5.1 权限粒度（以 Codex 为例）

| 沙箱模式 | 允许操作 | 典型场景 |
|---------|---------|---------|
| read-only | 只读命令（`ls`, `cat`, `rg`） | 审查、答疑 |
| workspace-write | 可写工作区内文件 + 受信命令 | 日常编码 |
| danger-full-access | 完全访问（网络、系统） | 需要网络/安装依赖 |

### 5.2 审批策略

- **untrusted**：只允许受信命令，其余请求用户审批。
- **on-request**：模型自行决定何时请求审批。
- **never**：从不请求审批，适合受信自动化环境。

### 5.3 安全建议

- 默认使用 `workspace-write`，避免模型意外触碰工作区外文件；
- 对不可信仓库先审查其 `AGENTS.md` / hooks 再放行；
- 不要在生产环境随意使用 `dangerously-bypass-approvals-and-sandbox`；
- 钩子（hooks）能拦截工具调用，可作额外的安全闸门。

---

## 6 Codex CLI 使用指南 {#6}

Codex CLI 是 OpenAI 开源的终端原生编程智能体，支持交互式 TUI 与脚本化执行。

### 6.1 安装与登录

```bash
npm install -g @openai/codex
codex login                 # 登录并授权
codex doctor                # 诊断安装、配置、鉴权、运行环境
```

### 6.2 交互式使用

```bash
codex "修复这个仓库里的所有 lint 错误"    # 启动交互式会话
codex                           # 进入 TUI，输入 /help 查看内置命令
```

### 6.3 非交互式执行

```bash
# 单次执行并退出（适合脚本/CI）
codex exec "为 data_loader.py 补充单元测试"

# 从 stdin 读取提示
cat prompt.md | codex exec -

# 指定模型与工作目录
codex exec -m gpt-5 "重构这个模块"
codex exec -C /path/to/repo "解释此仓库结构"

# 生成代码审查报告
codex review

# 恢复上次会话 / 指定会话
codex exec resume --last
codex resume --last
```

### 6.4 权限与沙箱

```bash
# 只读沙箱（默认对生成命令做审批）
codex -s read-only "帮我看看这段代码有什么问题"

# 允许写工作区
codex -s workspace-write "实现这个新功能"

# 完全访问（谨慎使用）
codex -s danger-full-access "pip install 依赖并运行测试"

# 审批策略：从不询问（受信环境）
codex -a never "运行测试并修复失败"
```

### 6.5 常用会话管理

```bash
codex resume              # 交互式选择恢复的会话
codex resume --last       # 继续最近一次会话
codex archive             # 归档旧会话
codex apply               # 将 agent 最新 diff 应用为 git apply
codex fork --last         # 从最近会话派生新会话
codex mcp add <name> <cmd> # 添加 MCP 服务器
```

### 6.6 配置

配置文件位于 `~/.codex/config.toml`，可用 `-c` 覆盖：

```bash
codex -c model="o4-mini" -c 'sandbox_permissions=["workspace-write"]' "..."
```

---

## 7 Claude Code 使用指南 {#7}

Claude Code 是 Anthropic 的终端原生编程智能体，深度集成 CLAUDE.md、skills、hooks 与 MCP。

### 7.1 安装与鉴权

```bash
npm install -g @anthropic-ai/claude-code
claude                 # 启动交互式会话（首次运行自动登录）
claude doctor          # 检查安装与配置健康
```

### 7.2 交互式使用

```bash
claude "解释并重构 src/utils.ts"
claude -c              # 继续当前目录最近一次会话
claude -r              # 恢复会话（交互选择或按 ID）
```

会话内常用命令：

```bash
/help                  # 查看所有斜杠命令
/model <model>         # 切换模型（如 opus / sonnet）
/context               # 查看当前上下文占用
/compact               # 手动压缩上下文
/resume                # 恢复历史会话
/agents                # 管理后台 agent
/mcp                   # 管理 MCP 服务器
/memory                # 查看自动记忆
```

### 7.3 非交互式执行（Print 模式）

```bash
# 单次问答并打印结果
claude -p "给这个函数写 JSDoc"

# 管道输入
cat bug_report.txt | claude -p "根据这个报告定位问题"

# 输出 JSON
claude -p --output-format json "列出所有 TODO"

# 指定模型与最大花费
claude -p --model sonnet --max-budget-usd 2 "..."

# 指定权限模式（如自动接受编辑）
claude -p --permission-mode acceptEdits "重构这个模块"
```

### 7.4 后台 Agent 与自动化

```bash
claude agents                    # 管理后台 agent
claude agents --name fix-tests --bg "修复失败的测试"   # 后台运行
claude -c --bg                   # 后台继续会话
```

### 7.5 项目级配置

- `CLAUDE.md`（项目根目录）：项目指令，自动注入；
- `~/.claude/settings.json`：全局配置（模型、权限、hooks、permission-mode）；
- `.claude/settings.json`：项目级配置；
- 自定义 skills / plugins：通过 `--plugin-dir` 或 `/plugin` 安装。

### 7.6 安全选项

```bash
claude --allowedTools "Bash(git *) Edit"    # 只允许白名单工具
claude --disallowedTools "WebSearch"         # 禁用某工具
claude --permission-mode plan                # 只规划不执行
claude --dangerously-skip-permissions        # 跳过全部权限（仅限外部沙箱）
```

---

## 8 工程角色：Harness / Loop / Evals Engineer {#8}

编程智能体的成熟催生了新的工程分工，其中 "Harness Engineer" 与 "Loop Engineer" 是 2025 年前后兴起的核心角色。

### 8.1 Harness Engineer（编排工程师）

负责构建和维护智能体的"骨架"——让 LLM 与外部世界可靠交互的那层基础设施。

核心职责：

- 设计 **agent loop**：规划、工具调用、观察、重试、终止条件的编排；
- 构建 **工具层**：文件读写、命令执行、搜索、MCP 接入的封装与错误处理；
- 实现 **会话与状态管理**：上下文打包、压缩、恢复、checkpoint；
- 设计 **沙箱与权限**：命令白名单、目录边界、审批策略、hooks 闸门；
- 提供 **可观测性**：tracing、日志、token 与成本统计、评估回放。

一句话：Harness Engineer 决定"智能体能做什么、边界在哪、出问题怎么恢复"。

### 8.2 Loop Engineer（循环工程师）

专注优化 agent loop 的"每一圈"效率与质量，让模型更少犯错、更快收敛。

核心职责：

- **提示工程**：优化系统提示、AGENTS.md/CLAUDE.md、工具描述，减少无效探索；
- **上下文优化**：选择注入哪些仓库信息、何时压缩、如何压缩，控制 token 与成本；
- **工具设计**：设计更好的工具接口、错误信息与重试策略，让模型"看得懂反馈"；
- **循环调优**：调整推理轮次上限、编辑粒度、验证与修复策略，提升任务成功率；
- **技能与记忆**：沉淀可复用 skills、模板与最佳实践，加速常见任务。

一句话：Loop Engineer 决定"每一圈循环的效率与最终质量"。

### 8.3 配套角色

| 角色 | 职责 |
|------|------|
| Evals Engineer | 构建评测集与指标，量化 agent 在不同任务上的成功率与退化 |
| Agent Product Engineer | 面向用户打磨交互、权限确认流与错误提示 |
| Red Team Engineer | 针对 prompt injection、越权执行做安全测试 |

这三个角色共同构成"能写、能测、能防"的完整编程智能体工程团队。

---

## 9 实践建议 {#9}

### 9.1 选型建议

- **交互式开发**：Claude Code 的 CLAUDE.md + skills 体系完善，适合日常开发；
- **脚本化/CI 自动化**：Codex CLI 的 `exec` / `review` 命令更适合流水线集成；
- **IDE 集成**：Cursor 等提供编辑器内体验；
- **两者都装**：多数场景两者互补，按项目信任度与工作流切换。

### 9.2 让智能体更可靠

- 为仓库编写清晰的 `AGENTS.md`，明确规范、命令与约束；
- 把测试命令写进文档，让 agent 主动验证而不是"猜"；
- 用小步提交 + `git diff` 审查，而非一次性大改；
- 关键修改前让 agent 先给计划（`--permission-mode plan`），人工确认后再执行；
- 记录失败案例，沉淀为 evals 与 skills，持续改进 loop。

### 9.3 安全底线

- 默认 `workspace-write` 沙箱，不轻易全放开；
- 对不可信仓库先审查其指令文件与 hooks；
- 生产/敏感环境禁用 `dangerously-bypass-approvals-and-sandbox`；
- 用权限白名单（`--allowedTools` / `--disallowedTools`）收紧工具边界；
- 定期 `codex doctor` / `claude doctor` 检查环境健康。

---

## 10 参考文献 {#10}

[1] OpenAI, *Codex CLI: Your AI coding agent in your terminal*, GitHub, 2025. https://github.com/openai/codex

[2] Anthropic, *Claude Code: An agentic coding tool*, Anthropic Docs, 2025.

[3] J. Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*, NeurIPS, 2024.

[4] C. E. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, 2024.

[5] N. Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS, 2023.

[6] S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR, 2023.

[7] Model Context Protocol, *MCP: An open protocol for connecting AI to tools*, 2024.

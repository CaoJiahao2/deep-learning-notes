# 智能体

> 让模型从「回答一次」变成「完成一件事」。

Agent = LLM 大脑 + 工具 + 记忆 + 循环。本章节按「核心机制 → 协作与运行时 → 垂直领域 → 评测与安全」组织：核心机制讲清楚单个 Agent 如何感知、规划、调用工具、记住历史；协作与运行时讲多智能体编排与生产级 Agent 运行环境；垂直领域覆盖编程、GUI、网页三类最受关注的 Agent；评测与安全回答「怎么衡量好坏、怎么防滥用」。

## 章节地图

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } **核心机制**

    ---

    单个 Agent 的解剖学。

    - [AI Agent 基础与核心循环](./fundamentals.md) — 感知-规划-行动-反思循环
    - [工具使用与 MCP](./tools-and-mcp.md) — Function Calling、Model Context Protocol
    - [Agent 记忆机制](./memory.md) — 短期/长期、向量/笔记式、MemGPT
    - [Agentic RAG](./agentic-rag.md) — Agent 主导的检索与查询改写

-   :material-account-group:{ .lg .middle } **协作与运行时**

    ---

    从单兵作战到生产系统。

    - [多智能体协作](./multi-agent.md) — AutoGen、CrewAI、LangGraph
    - [Agent Runtime：OpenClaw 与 Hermes Agent](./agent-runtime.md) — Gateway/Runtime 架构、持久记忆、沙箱

-   :material-rocket-launch:{ .lg .middle } **垂直领域 Agent**

    ---

    最先落地的三类 Agent。

    - [编程智能体：架构与使用](./coding-agents.md) — Agent Loop、沙箱权限、Codex CLI
    - [GUI Agent 与 Computer Use](./gui-agent.md) — Operator、AutoGLM、屏幕 grounding
    - [网页智能体与浏览 Agent](./web-agents.md) — DOM/Accessibility、WebVoyager

-   :material-shield-check:{ .lg .middle } **评测与安全**

    ---

    度量与防线。

    - [Agent 评测与基准](./agent-evaluation.md) — AgentBench、GAIA、SWE-bench、OSWorld
    - [Agent 安全与提示注入](./agent-security.md) — 直接/间接注入、沙箱边界、OWASP

</div>

## 学习建议

1. **从核心循环开始**：[AI Agent 基础](./fundamentals.md) 的「感知-规划-行动-反思」循环是全章节的骨架，其余所有主题都是对这个循环某一环的深化。
2. **动手搭一个最小 Agent**：用 Function Calling 实现「检索 + 计算 + 写文件」三工具的小循环，再对照 [工具使用与 MCP](./tools-and-mcp.md) 理解协议标准化的重要性。
3. **安全与能力同步学**：每学一类垂直 Agent，同步看 [Agent 安全](./agent-security.md) 中对应的攻击面，间接注入是生产环境最大的风险。
4. **用评测驱动迭代**：[Agent 评测](./agent-evaluation.md) 中的基准（如 SWE-bench、WebArena）可以直接作为自己项目的回归测试集。

## 延伸阅读

- [提示工程](../llm-mllm/prompt-engineering.md) — Agent 的上下文工程底座
- [多模态推理模型 LMRM 综述](../llm-mllm/multimodal-reasoning-survey.md) — GUI Agent 的模型侧演进
- [LLM 应用工程（LLMOps）](../deployment/llm-ops.md) — Agent 上线的工程化配套

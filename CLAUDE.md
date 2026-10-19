# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chinese-language deep learning knowledge base (深度学习笔记) built with **MkDocs + Material** and **KaTeX** math rendering. All content lives in Markdown under `docs/`, organized by topic domain. Content is written in Chinese with English technical terms preserved. The site auto-deploys to GitHub Pages on push to `main`.

## Commands

```bash
# Local preview (live math rendering)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # mkdocs, mkdocs-material, pymdown-extensions, jieba
npm ci                             # KaTeX for the math linter (locked versions)
mkdocs serve                       # http://127.0.0.1:8000

# Math formula lint — scans docs/ for $..$ and $$..$$, fails on any non-compilable LaTeX.
# Pass specific files/dirs to check only what changed: node scripts/check_math.mjs docs/llm-mllm/x.md
node scripts/check_math.mjs        # or: npm run check:math

# Nav/document consistency — every docs/**/*.md must be in mkdocs.yml nav: and vice versa
node scripts/check_nav.mjs         # or: npm run check:nav

# Strict build (CI uses this; any warning fails)
mkdocs build --strict
```

CI (`.github/workflows/ci.yml` on PR/push, `.github/workflows/deploy.yml` on main) runs: `pip install -r requirements.txt` → `npm ci` → `node scripts/check_math.mjs` → `node scripts/check_nav.mjs` → `mkdocs build --strict` → deploy to GitHub Pages. Run both linters and the strict build before committing; a broken formula breaks CI.

## Architecture

- **`docs/`** — all content. Subdirectories are topic domains: `fundamentals/`, `architectures/`, `training/`, `llm-mllm/`, `agents/`, `embodied-ai/`, `deployment/`, `resources/`. `index.md` is the site home page.
- **`mkdocs.yml`** — the source of truth for navigation. `nav:` maps page titles to files, grouped into sub-sections per chapter. **When adding a new doc, add it to `nav:` here** and to the section overview page (`docs/<section>/index.md`, 章节地图卡片).
- **`scripts/check_math.mjs`** — KaTeX linter. Walks `docs/` for `.md` files, masks fenced code blocks and inline code spans, extracts block `$$...$$` then inline `$...$`, and renders each with `throwOnError: true`. Reports the real `file:line`, exits 1 on any failure. Accepts optional file/dir paths to check only what changed; with no args it scans all of `docs/` (what CI runs).
- **`scripts/check_nav.mjs`** — nav consistency linter: fails when a `nav:` entry points at a missing file, when a `docs/**/*.md` is absent from `nav:`, or when a filename is not kebab-case. `mkdocs build --strict` does not catch the "orphan page" case (only INFO), so this runs alongside it.
- **`docs/javascripts/katex.js`** — runtime renderer; registers `$$...$$`, `\\[...\\]`, `$...$`, `\\(...\\)` delimiters with Material's `document$` stream.
- **`asserts/`** — images (referenced by relative paths).
- **`site/`** — build output (gitignored).

## Conventions (from CONTRIBUTING.md)

- Body text in Chinese, technical terms in English.
- Block math uses `$$...$$`, inline uses `$...$`. Both must compile under KaTeX.
- Code blocks must tag a language (```python, ```bash, ```text).
- Each doc ends with a 参考文献 (references) section — paper title + year.
- Filenames: lowercase + hyphen (e.g. `fine-tuning.md`).
- Doc structure: 概述 → 核心原理 → 数学推导 → 实践建议 → 参考文献.

## 新增内容检查 (Checking new/changed content)

Only check the math you actually added — judge it against the md syntax rules below, then run the strict build. No need to rescan the whole corpus each time.

```bash
# 1) check only the files you touched (each path may be a file or a directory)
node scripts/check_math.mjs docs/llm-mllm/rlhf.md docs/fundamentals/backprop.md
# find changed files:  git diff --name-only HEAD~1 -- docs/

# 2) real gate — the nav checker + strict build
node scripts/check_nav.mjs
mkdocs build --strict
```

**Syntax rules the linter enforces** (scripts/check_math.mjs is regex-based — these are hard limits):

- Only `$$...$$` (block) and `$...$` (inline) are linted. The runtime renderer also accepts `\[...\]` / `\(...\)`, but the linter does NOT check those — never use them for new formulas.
- Inline `$...$` must stay on one line and contain no literal `$` (so `\$` breaks it too). Spanning a line means it is silently skipped by the linter — a broken formula that CI won't catch.
- `$` must be paired. An unpaired `$` in prose is silently skipped by the linter but garbles rendering.
- `$$...$$` closes at the first `$$` — block math cannot contain `$$` inside.
- Fenced code blocks and inline code spans are skipped, but do not rely on that for real formulas.

**Workflow**: edit → `node scripts/check_math.mjs <changed files>` → `node scripts/check_nav.mjs` → `mkdocs build --strict`. A non-compilable formula fails CI.

## TODO 目录处理流程 (Processing the TODO directory)

`TODO/` 存放待处理的 Markdown 文档（该目录在 `.gitignore` 中，属于本地草稿区，不会被提交）。每次处理流程如下：

1. 阅读 `TODO/*.md` 的内容，判断主题归属，将内容整合进网站合适的章节（例如 LLM 后训练 → `docs/llm-mllm/`、多模态推理 → `docs/llm-mllm/` 等）。
2. 整合时遵守本文档的公式规范：新公式一律使用 `$$...$$`（块级）与 `$...$`（行内）。TODO 文档若使用 `\[...\]` / `\(...\)`，必须转换为 `$$...$$` / `$...$`（linter 不检查 `\[...\]`）。
3. 文档以「参考文献」章节结尾（论文标题 + 年份）。
4. 在 `mkdocs.yml` 的 `nav:`（放入正确子分组）与对应章节总览页 `docs/<section>/index.md` 中登记新文档。
5. 用 linter 与严格构建校验改动：`node scripts/check_math.mjs <changed files>` → `mkdocs build --strict`。
6. 处理完成后直接删除 `TODO/` 下的 md 文档（删除文件本身）。
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chinese-language deep learning knowledge base (深度学习笔记) built with **MkDocs + Material** and **KaTeX** math rendering. All content lives in Markdown under `docs/`, organized by topic domain. Content is written in Chinese with English technical terms preserved. The site auto-deploys to GitHub Pages on push to `main`.

## Commands

```bash
# Local preview (live math rendering)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # mkdocs, mkdocs-material, pymdown-extensions
npm install                        # KaTeX for the math linter
mkdocs serve                       # http://127.0.0.1:8000

# Math formula lint — scans docs/ for $..$ and $$..$$, fails on any non-compilable LaTeX.
# Pass specific files/dirs to check only what changed: node scripts/check_math.mjs docs/llm-mllm/x.md
node scripts/check_math.mjs        # or: npm run check:math

# Strict build (CI uses this; any warning fails)
mkdocs build --strict
```

CI (`.github/workflows/deploy.yml`) runs: `pip install -r requirements.txt` → `npm ci` → `node scripts/check_math.mjs` → `mkdocs build --strict --site-dir _site` → deploy to GitHub Pages. Run the math linter and strict build before committing; a broken formula breaks CI.

## Architecture

- **`docs/`** — all content. Subdirectories are topic domains: `getting-started/`, `fundamentals/`, `architectures/`, `training/`, `llm-mllm/`, `deployment/`, `resources/`. `index.md` is the site home page.
- **`mkdocs.yml`** — the source of truth for navigation. `nav:` maps page titles to files. **When adding a new doc, add it to `nav:` here** (and to `docs/index.md` per contributing checklist).
- **`scripts/check_math.mjs`** — KaTeX linter. Walks `docs/` for `.md` files, extracts block `$$...$$` then inline `$...$` (skipping ranges already consumed by block spans), and renders each with `throwOnError: true`. Reports `file:line`, exits 1 on any failure. Accepts optional file/dir paths to check only what changed; with no args it scans all of `docs/` (what CI runs).
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

# 2) real gate — the linter can't catch nav/heading warnings
mkdocs build --strict
```

**Syntax rules the linter enforces** (scripts/check_math.mjs is regex-based — these are hard limits):

- Only `$$...$$` (block) and `$...$` (inline) are linted. The runtime renderer also accepts `\[...\]` / `\(...\)`, but the linter does NOT check those — never use them for new formulas.
- Inline `$...$` must stay on one line and contain no literal `$` (so `\$` breaks it too). Spanning a line means it is silently skipped by the linter — a broken formula that CI won't catch.
- `$` must be paired. An unpaired `$` in prose is silently skipped by the linter but garbles rendering.
- `$$...$$` closes at the first `$$` — block math cannot contain `$$` inside.
- The linter does NOT skip fenced code blocks — keep `$` out of code samples.

**Workflow**: edit → `node scripts/check_math.mjs <changed files>` → `mkdocs build --strict`. A non-compilable formula fails CI.
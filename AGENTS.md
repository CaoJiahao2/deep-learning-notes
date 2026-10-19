# Repository Guidelines

## Project Structure

This is a Chinese-language deep learning knowledge base built with **MkDocs Material** and **KaTeX**. The repository is documentation, tooling, and site config — no application code or traditional test suite.

- `docs/` — All Markdown content, grouped by domain (`architectures/`, `training/`, `llm-mllm/`, `agents/`, `embodied-ai/`, `deployment/`, `fundamentals/`, `resources/`).
- `mkdocs.yml` — Site config and the source of truth for navigation (`nav:`).
- `scripts/check_math.mjs` — KaTeX linter that validates all LaTeX in `docs/` (skips code blocks/code spans).
- `scripts/check_nav.mjs` — validates that `mkdocs.yml` `nav:` ↔ `docs/**/*.md` match both ways and filenames are kebab-case.
- `asserts/` — Images and static assets.
- `TODO/` — Local, git-ignored drafts awaiting integration (never committed).
- `site/` — Build output (gitignored).

## Build & Development Commands

```bash
# Setup (Python >= 3.9, Node >= 18)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && npm ci

# Live preview at http://127.0.0.1:8000
mkdocs serve

# Lint math formulas
node scripts/check_math.mjs                       # all docs
node scripts/check_math.mjs docs/path/file.md     # specific file

# Check nav ↔ docs consistency / filename conventions
node scripts/check_nav.mjs

# Strict production build (warnings fail; used by CI)
mkdocs build --strict
```

CI runs: install → `check_math.mjs` → `check_nav.mjs` → `mkdocs build --strict` → deploy to GitHub Pages. Always run the linters and strict build locally before pushing.

## Style & Naming Conventions

- **Language**: Chinese prose; keep technical terms in English (Transformer、fine-tuning、RLHF).
- **Filenames/directories**: lowercase with hyphens (`fine-tuning.md`, `llm-mllm/`).
- **Math**: block uses `$$...$$`; inline uses `$...$`. Both must compile under KaTeX. Do not use `\[...\]` or `\(...\)` — the linter ignores them.
- **Code blocks**: always tag a language (```` ```python ````, ```` ```bash ````, ```` ```text ````).
- **Indentation**: 2 spaces in YAML/Markdown; 4 spaces in code blocks.
- **Document structure**: 概述 → 核心原理 → 数学推导 → 实践建议 → 参考文献.

## Testing & Validation

There is no unit-test framework. Validation is:

1. `node scripts/check_math.mjs` — catches non-compilable LaTeX.
2. `node scripts/check_nav.mjs` — catches pages missing from `nav:` (or nav entries pointing at missing files) and bad filenames.
3. `mkdocs build --strict` — catches broken links and Markdown warnings. It does **not** fail on "page exists but is not in `nav:`" (only INFO), which is why `check_nav.mjs` exists.

All three must pass before submitting a PR.

## Commit & Pull Request Guidelines

- **Commits**: Chinese, concise and descriptive. Use `新增 <topic>` for additions, `更新 <topic>` for revisions.
- **Branches**: `feature/your-topic` off `main`.
- **PR checklist**:
  - Register new pages in `mkdocs.yml` `nav:` (inside the correct sub-group) and add them to the section overview page (`docs/<section>/index.md` 章节地图).
  - End every new document with 参考文献 (paper title + year).
  - Run both linters on changed files and `mkdocs build --strict`.
  - Describe the change and link related issues.

## TODO Integration Workflow

When processing drafts in `TODO/`: place content in the correct `docs/` subdirectory, convert `\[...\]`/`\(...\)` formulas to `$$...$$`/`$...$`, register in `mkdocs.yml`, validate with linter + strict build, then delete the draft from `TODO/`.

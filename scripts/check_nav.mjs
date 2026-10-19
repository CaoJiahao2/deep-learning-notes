#!/usr/bin/env node
// 导航一致性校验脚本：
// 1) mkdocs.yml 的 nav 中引用的 .md 必须存在；
// 2) docs/ 下的每个 .md 都必须登记在 nav 中（防止「文档写了但站点看不到」）；
// 3) 文件名必须是小写字母/数字/连字符（kebab-case）。
//
// mkdocs build --strict 只能发现「nav 指向不存在的文件」，
// 对「文件存在但未登记 nav」只打印 INFO，不会失败，因此需要本脚本兜底。
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, basename } from "node:path";

const ROOT = process.cwd();
const CONFIG = join(ROOT, "mkdocs.yml");
const DOCS = join(ROOT, "docs");

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}

const config = readFileSync(CONFIG, "utf8");
const navBlock = config.split(/\n(?=nav:\s*$)/m)[1] ?? "";
const navPaths = new Set(
  [...navBlock.matchAll(/:\s*([A-Za-z0-9_./-]+\.md)\s*$/gm)].map((m) => m[1])
);

const actual = new Set(
  walk(DOCS).map((p) => relative(DOCS, p).split("\\").join("/"))
);

let failed = false;

const missing = [...navPaths].filter((p) => !actual.has(p)).sort();
if (missing.length) {
  failed = true;
  console.error("✗ nav 引用了不存在的文档：");
  for (const p of missing) console.error(`  - docs/${p}`);
}

const orphans = [...actual].filter((p) => !navPaths.has(p)).sort();
if (orphans.length) {
  failed = true;
  console.error("✗ 以下文档未登记到 mkdocs.yml 的 nav（站点上不可见）：");
  for (const p of orphans) console.error(`  - docs/${p}`);
}

const badNames = [...actual]
  .filter((p) => !/^[a-z0-9]+(-[a-z0-9]+)*\.md$/.test(basename(p)))
  .sort();
if (badNames.length) {
  failed = true;
  console.error("✗ 文件名不符合 kebab-case 规范：");
  for (const p of badNames) console.error(`  - docs/${p}`);
}

if (failed) process.exit(1);
console.log(
  `✓ nav 与文档一致：${navPaths.size} 个导航条目，${actual.size} 个文档，文件名规范`
);

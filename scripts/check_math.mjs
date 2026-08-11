#!/usr/bin/env node
// KaTeX 数学公式校验脚本：
// 扫描 docs/ 下所有 .md 文件，提取 $...$ 与 $$...$$ 中的 LaTeX，
// 用 KaTeX 解析校验，任何无法编译的公式都会导致 CI 失败。
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import katex from "katex";

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}

// 提取公式。先处理块级 $$...$$，再处理行内 $...$。
function extractMath(src) {
  const math = [];
  // 块级
  const blockRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  const blockSpan = [];
  while ((m = blockRe.exec(src)) !== null) {
    math.push({ tex: m[1], display: true, file: null, kind: "block" });
    blockSpan.push([m.index, m.index + m[0].length]);
  }
  // 行内（跳过已被块级占用的区间）
  const inlineRe = /\$(?!\$)([^$\n]+?)\$/g;
  while ((m = inlineRe.exec(src)) !== null) {
    const conflict = blockSpan.some(
      ([s, e]) => m.index >= s && m.index < e
    );
    if (!conflict) math.push({ tex: m[1], display: false, kind: "inline" });
  }
  return math;
}

// 支持传参：node check_math.mjs <file|dir> ...
// 未传参时默认扫描 docs/ 全部 .md（CI 行为不变）
const targets = process.argv.slice(2);
const files = [];
for (const p of (targets.length ? targets : ["docs"])) {
  let st;
  try { st = statSync(p); } catch { console.error(`✗ 路径不存在: ${p}`); process.exit(1); }
  if (st.isDirectory()) files.push(...walk(p));
  else if (p.endsWith(".md")) files.push(p);
  else console.warn(`⚠ 跳过非 .md 文件: ${p}`);
}

let failures = 0;
let total = 0;
let fileCount = 0;

for (const file of files) {
  const src = readFileSync(file, "utf8");
  const items = extractMath(src);
  if (items.length === 0) continue;
  fileCount++;
  for (const item of items) {
    total++;
    try {
      katex.renderToString(item.tex, { displayMode: item.display, throwOnError: true });
    } catch (e) {
      failures++;
      const line = src.slice(0, src.indexOf(item.tex)).split("\n").length;
      console.error(`✗ ${file}:${line} [${item.kind}] ${item.tex.slice(0, 80)}`);
      console.error(`    → ${e.message.split("\n")[0]}`);
    }
  }
}

console.log(`✓ 扫描 ${fileCount} 个文件，共 ${total} 条公式，失败 ${failures} 条`);
if (failures > 0) {
  console.error("存在无法渲染的数学公式，请修复后重试。");
  process.exit(1);
}

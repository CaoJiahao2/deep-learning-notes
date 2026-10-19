#!/usr/bin/env node
// KaTeX 数学公式校验脚本：
// 扫描 docs/ 下所有 .md 文件，提取 $...$ 与 $$...$$ 中的 LaTeX，
// 用 KaTeX 解析校验，任何无法编译的公式都会导致 CI 失败。
//
// 说明：
// - 跳过 fenced code block（``` / ~~~）与行内 code span 中的 `$`，避免代码示例误报；
// - 每条公式记录其真实行号（基于原始文本的偏移量），重复公式也能定位；
// - 只检查 $$...$$ 与 $...$；\[...\]、\(...\) 不在检查范围内（请改用 $$/$）。
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

// 把代码区域（fenced code block / 行内 code span）替换为空格，保留换行以便计算行号。
function maskCode(src) {
  const lines = src.split("\n");
  let fence = null; // 当前 fence 标记：``` 或 ~~~
  const masked = lines.map((line) => {
    const m = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (fence) {
      if (m && m[1][0] === fence[0] && m[1].length >= fence.length) fence = null;
      return " ".repeat(line.length);
    }
    if (m) {
      fence = m[1];
      return " ".repeat(line.length);
    }
    return line;
  });
  let text = masked.join("\n");
  // 行内 code span（同一行内成对的等长反引号）
  text = text.replace(/(`+)([^`\n]*?)\1/g, (s) => " ".repeat(s.length));
  return text;
}

function lineOf(src, index) {
  let line = 1;
  for (let i = 0; i < index && i < src.length; i++) {
    if (src.charCodeAt(i) === 10) line++;
  }
  return line;
}

// 提取公式。先处理块级 $$...$$，再处理行内 $...$。
function extractMath(src) {
  const masked = maskCode(src);
  const math = [];
  const blockSpan = [];
  const blockRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  while ((m = blockRe.exec(masked)) !== null) {
    math.push({
      tex: m[1],
      display: true,
      kind: "block",
      line: lineOf(masked, m.index),
    });
    blockSpan.push([m.index, m.index + m[0].length]);
  }
  const inlineRe = /\$(?!\$)([^$\n]+?)\$/g;
  while ((m = inlineRe.exec(masked)) !== null) {
    const conflict = blockSpan.some(([s, e]) => m.index >= s && m.index < e);
    if (!conflict) {
      math.push({
        tex: m[1],
        display: false,
        kind: "inline",
        line: lineOf(masked, m.index),
      });
    }
  }
  return math;
}

// 支持传参：node check_math.mjs <file|dir> ...
// 未传参时默认扫描 docs/ 全部 .md（CI 行为不变）
const targets = process.argv.slice(2);
const files = [];
for (const p of targets.length ? targets : ["docs"]) {
  let st;
  try {
    st = statSync(p);
  } catch {
    console.error(`✗ 路径不存在: ${p}`);
    process.exit(1);
  }
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
      katex.renderToString(item.tex, {
        displayMode: item.display,
        throwOnError: true,
      });
    } catch (e) {
      failures++;
      console.error(
        `✗ ${file}:${item.line} [${item.kind}] ${item.tex.slice(0, 80)}`
      );
      console.error(`    → ${e.message.split("\n")[0]}`);
    }
  }
}

console.log(`✓ 扫描 ${fileCount} 个文件，共 ${total} 条公式，失败 ${failures} 条`);
if (failures > 0) {
  console.error("存在无法渲染的数学公式，请修复后重试。");
  process.exit(1);
}

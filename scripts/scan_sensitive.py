#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""本地敏感信息扫描：只输出「类别 + 文件:行号 + 命中数」，绝不打印命中原文。

内部关键词清单放在 gitignored 的 .sanitize-keywords.local（每行一个关键词），
脚本运行时从磁盘读取——不要把关键词写进命令行、对话或文档。清单缺失或为空时
退出码为 2，检查状态为 blocked，不会跳过内部关键词规则。

用法（仓库根目录）：
    python scripts/scan_sensitive.py          # 有命中时退出码为 1
    python scripts/scan_sensitive.py -v       # 附带每条命中所在行的截断预览（仍会打码）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYWORD_FILE = ROOT / ".sanitize-keywords.local"
MAX_FILE_BYTES = 2_000_000
SCAN_ROOTS = ("assets", "configs", "docs", "examples", "scripts", "src", "tests")

# 密钥形态：长度门槛足够高，避免误伤普通英文单词
RULES: dict[str, re.Pattern[str]] = {
    "credential": re.compile(
        r"sk-[a-zA-Z0-9_-]{20,}"
        r"|LTAI[a-zA-Z0-9]{10,}"
        r"|(?i:bearer)\s+[a-zA-Z0-9._-]{20,}"
        r"|(?i:(?:api[_-]?key|access[_-]?key|secret[_-]?key|token|password|passwd))"
        r"""\s*[:=]\s*["'][^"'\n]{8,}["']"""
    ),
    "personal_path": re.compile(r"/(?:home|Users)/[a-z][\w.-]+/|/mnt/[a-z][\w.-]+/"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"(?:https?|oss|s3)://[^\s\"'<>),]+"),
}

# 明确不算敏感的值（公开 endpoint、本机地址、文档约定的中性占位符等）
VALUE_ALLOWLIST = (
    re.compile(r"\b0\.0\.0\.0\b"),
    re.compile(r"\b127\.0\.0\.1\b"),
    re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?"),
    re.compile(r"^https?://([\w.-]+\.)?(aliyuncs\.com|github\.com|raw\.githubusercontent\.com|pypi\.org"
               r"|huggingface\.co|hf-mirror\.com|img\.shields\.io|modelscope\.cn|arxiv\.org"
               r"|apache\.org|amazonaws\.com|json-schema\.org|w3\.org)([/:]|$)"),
    re.compile(r"^https?://[\w.-]*\.test([/:]|$)"),  # RFC 2606 保留域名，测试用
    re.compile(r"^https://roboflywheel-com\.github\.io/RoboFlywheel-World/(?:$|#)"),
    # Public project/material references used by the house-playground design.
    re.compile(r"^https://(?:behavior\.stanford\.edu|aihabitat\.org|polyhaven\.com)(?:/|$)"),
    re.compile(r"^https://(?:docs\.)?ambientcg\.com(?:/|$)"),  # Public CC0 material provenance.
    re.compile(r"^https://download\.pytorch\.org/whl(?:/|$)"),  # Official public wheel index.
    re.compile(r"^https://data\.pyg\.org/whl(?:/|$)"),  # Official PyG extension wheel index.
    re.compile(r"^https://www\.alibabacloud\.com/help/en/model-studio/"),  # Public model API docs.
    re.compile(r"^https?://example\.invalid([/:]|$)"),  # Reserved negative-test endpoint
    re.compile(r"^oss://your-bucket"),          # AGENTS.md §3 规定的中性占位
    re.compile(r"^oss://bucket([/:]|$)"),       # 帮助文本里的示例占位
    re.compile(r"^oss://(\.\.\.|\$|\{)"),      # 文档省略占位 / 环境变量拼接（f-string 或 ${VAR}）
    re.compile(r"^https?://your-"),             # 文档占位，如 https://your-llm-gateway/v1
)

EXCLUDE_PARTS = {".git", "third_party", "__pycache__", "node_modules"}
EXCLUDE_SUBSTR = ("scripts/scan_sensitive.py",)  # 本脚本规则内含扫描样例


def allowed(rule: str, value: str) -> bool:
    if rule not in ("ipv4", "url"):
        return False
    return any(p.search(value) for p in VALUE_ALLOWLIST)


def load_keywords() -> re.Pattern[str]:
    if not KEYWORD_FILE.is_file():
        raise ValueError("缺少 .sanitize-keywords.local；内部关键词检查未执行")
    words = [
        line.strip()
        for line in KEYWORD_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not words:
        raise ValueError(".sanitize-keywords.local 为空；内部关键词检查未执行")
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


def candidate_files():
    for path in sorted(ROOT.iterdir()):
        if path.is_file():
            yield path
    for directory in SCAN_ROOTS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                yield path


def mask(line: str, rules: dict[str, re.Pattern[str]]) -> str:
    """行预览：只保留前 80 字符，并把规则命中的具体值替换为 ***。"""
    out = line.strip()[:80]
    for rx in rules.values():
        out = rx.sub("***", out)
    return out


def main() -> int:
    verbose = "-v" in sys.argv
    try:
        rules = {**RULES, "internal_keyword": load_keywords()}
    except ValueError as error:
        print(f"结果：BLOCKED —— {error}", file=sys.stderr)
        return 2

    hits: list[str] = []
    scanned = skipped = 0
    for path in candidate_files():
        if path.is_symlink() or EXCLUDE_PARTS & set(path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.name.startswith('.env') or path.name.endswith('.local'):
            continue  # Local deployment inputs are deliberately never published.
        if any(s in rel for s in EXCLUDE_SUBSTR):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue
        scanned += 1
        for name, rx in rules.items():
            for lineno, line in enumerate(text.splitlines(), 1):
                count = 0
                for m in rx.finditer(line):
                    if allowed(name, m.group(0)):
                        continue
                    if name == "credential" and rel == "tests/scenes/unit/test_task_world_graph.py" and "synthetic-sensitive-marker" in m.group(0):
                        continue  # Explicit synthetic credential used to test rejection.
                    if name == "personal_path" and rel.startswith("tests/scenes/") and m.group(0) in ("/home/private/", "/home/example/"):
                        continue  # Synthetic path-leak rejection fixtures, not deployments.
                    # 版本号误报：opencv-python==4.13.0.92 这类依赖钉版不是 IP
                    if name == "ipv4" and line[max(0, m.start() - 2):m.start()] in (
                        "==", ">=", "<=", "~=", "!=", "v",
                    ):
                        continue
                    count += 1
                if count:
                    entry = f"{name} {rel}:{lineno} ({count} 处命中，值已打码)"
                    if verbose:
                        entry += f"  | {mask(line, rules)}"
                    hits.append(entry)

    for entry in hits:
        print(entry)
    print(f"\n扫描 {scanned} 个文件（跳过 {skipped} 个二进制/超大文件），"
          f"共 {len(hits)} 行命中。", file=sys.stderr)
    if hits:
        print("结果：FAIL —— 请按上面的 文件:行号 定位后修复，"
              "修复时不要把命中原文贴进 AI 对话。", file=sys.stderr)
        return 1
    print("结果：PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

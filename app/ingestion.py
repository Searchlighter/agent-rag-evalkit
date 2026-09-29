"""将本地 JSONL/CSV 文件转换为统一的评测样本字典。"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any


def load_jsonl_cases(path: str | Path) -> list[dict[str, Any]]:
    """逐行读取 UTF-8 JSONL，并在错误中保留准确行号。"""
    return parse_jsonl_cases(Path(path).read_text(encoding="utf-8"))


def parse_jsonl_cases(content: str) -> list[dict[str, Any]]:
    """解析浏览器或文件读取器提供的 JSONL 文本。"""
    cases: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(content.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 第 {line_number} 行格式错误：{error.msg}") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是 JSON 对象")
        cases.append(parsed)
    return cases


def load_csv_cases(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 CSV; multi-value fields use | as delimiter."""
    return parse_csv_cases(Path(path).read_text(encoding="utf-8-sig"))


def parse_csv_cases(content: str) -> list[dict[str, Any]]:
    """解析 CSV 文本，多值字段使用竖线分隔。"""
    cases: list[dict[str, Any]] = []
    reader = csv.DictReader(StringIO(content.lstrip("\ufeff"), newline=""))
    if not reader.fieldnames or "question" not in reader.fieldnames:
        raise ValueError("CSV 必须包含 question 表头")
    for line_number, row in enumerate(reader, start=2):
        question = (row.get("question") or "").strip()
        if not question:
            raise ValueError(f"CSV 第 {line_number} 行缺少 question")
        cases.append({
            "id": (row.get("id") or "").strip() or f"csv-{line_number}",
            "question": question,
            "expected_answers": _split_multi_value(row.get("expected_answers")),
            "expected_evidence": _split_multi_value(row.get("expected_evidence")),
            "tags": _split_multi_value(row.get("tags")),
        })
    return cases


def parse_dataset_content(filename: str, content: str) -> list[dict[str, Any]]:
    """按上传文件扩展名选择解析器并拒绝不支持的格式。"""
    suffix = Path(filename.strip()).suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl_cases(content)
    if suffix == ".csv":
        return parse_csv_cases(content)
    raise ValueError("仅支持 .jsonl 或 .csv 文件")


def _split_multi_value(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]

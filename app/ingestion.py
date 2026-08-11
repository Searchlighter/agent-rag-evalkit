"""将本地 JSONL/CSV 文件转换为统一的评测样本字典。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl_cases(path: str | Path) -> list[dict[str, Any]]:
    """逐行读取 UTF-8 JSONL，并在错误中保留准确行号。"""
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 第 {line_number} 行格式错误：{error.msg}") from error
    return cases


def load_csv_cases(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 CSV; multi-value fields use | as delimiter."""
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            question = (row.get("question") or "").strip()
            if not question:
                raise ValueError(f"CSV line {line_number}: question is required")
            cases.append({
                "id": (row.get("id") or "").strip() or f"csv-{line_number}",
                "question": question,
                "expected_answers": _split_multi_value(row.get("expected_answers")),
                "expected_evidence": _split_multi_value(row.get("expected_evidence")),
                "tags": _split_multi_value(row.get("tags")),
            })
    return cases


def _split_multi_value(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]

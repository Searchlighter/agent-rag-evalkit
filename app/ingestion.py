"""将本地 JSONL/CSV 文件转换为统一的评测样本字典。"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any


class DatasetUploadValidationError(ValueError):
    """包含可供页面逐项展示的数据集校验问题。"""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        super().__init__(f"数据集校验失败，共 {len(issues)} 个问题")


def load_jsonl_cases(path: str | Path) -> list[dict[str, Any]]:
    """逐行读取 UTF-8 JSONL，并在错误中保留准确行号。"""
    return parse_jsonl_cases(Path(path).read_text(encoding="utf-8"))


def parse_jsonl_cases(content: str) -> list[dict[str, Any]]:
    """解析浏览器或文件读取器提供的 JSONL 文本。"""
    cases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    id_lines: dict[str, list[int]] = {}
    for line_number, raw_line in enumerate(content.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(_issue("invalid_json", f"JSON 格式错误：{error.msg}", line_number))
            continue
        if not isinstance(parsed, dict):
            issues.append(_issue("invalid_record", "每行必须是 JSON 对象", line_number))
            continue
        shape_issues = _validate_json_case_shape(parsed, line_number)
        if shape_issues:
            issues.extend(shape_issues)
            continue
        cases.append(parsed)
        case_id = str(parsed.get("id") or "").strip()
        if case_id:
            id_lines.setdefault(case_id, []).append(line_number)
    issues.extend(_duplicate_id_issues(id_lines))
    if not cases and not issues:
        issues.append(_issue("empty_dataset", "文件中没有可导入的评测样本"))
    _raise_if_issues(issues)
    return cases


def load_csv_cases(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 CSV; multi-value fields use | as delimiter."""
    return parse_csv_cases(Path(path).read_text(encoding="utf-8-sig"))


def parse_csv_cases(content: str) -> list[dict[str, Any]]:
    """解析 CSV 文本，多值字段使用竖线分隔。"""
    cases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    id_lines: dict[str, list[int]] = {}
    reader = csv.DictReader(StringIO(content.lstrip("\ufeff"), newline=""))
    if not reader.fieldnames or "question" not in reader.fieldnames:
        raise DatasetUploadValidationError([
            _issue("missing_field", "CSV 必须包含 question 表头", 1, "question")
        ])
    for line_number, row in enumerate(reader, start=2):
        question = (row.get("question") or "").strip()
        if not question:
            issues.append(
                _issue("missing_field", "question 不能为空", line_number, "question")
            )
            continue
        case_id = (row.get("id") or "").strip() or f"csv-{line_number}"
        cases.append({
            "id": case_id,
            "question": question,
            "expected_answers": _split_multi_value(row.get("expected_answers")),
            "expected_evidence": _split_multi_value(row.get("expected_evidence")),
            "tags": _split_multi_value(row.get("tags")),
        })
        id_lines.setdefault(case_id, []).append(line_number)
    issues.extend(_duplicate_id_issues(id_lines))
    if not cases and not issues:
        issues.append(_issue("empty_dataset", "文件中没有可导入的评测样本"))
    _raise_if_issues(issues)
    return cases


def parse_dataset_content(filename: str, content: str) -> list[dict[str, Any]]:
    """按上传文件扩展名选择解析器并拒绝不支持的格式。"""
    suffix = Path(filename.strip()).suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl_cases(content)
    if suffix == ".csv":
        return parse_csv_cases(content)
    raise DatasetUploadValidationError([
        _issue("unsupported_format", "仅支持 .jsonl 或 .csv 文件")
    ])


def _validate_json_case_shape(
    raw: dict[str, Any], line_number: int
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(raw.get("question"), str) or not raw["question"].strip():
        issues.append(
            _issue("missing_field", "question 不能为空", line_number, "question")
        )
    for field_name in ("expected_answers", "expected_evidence", "tags"):
        value = raw.get(field_name)
        if value is not None and (
            not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
        ):
            issues.append(
                _issue(
                    "invalid_field",
                    f"{field_name} 必须是字符串数组",
                    line_number,
                    field_name,
                )
            )
    if "metadata" in raw and not isinstance(raw["metadata"], dict):
        issues.append(
            _issue("invalid_field", "metadata 必须是对象", line_number, "metadata")
        )
    return issues


def _duplicate_id_issues(id_lines: dict[str, list[int]]) -> list[dict[str, Any]]:
    return [
        {
            **_issue(
                "duplicate_id",
                f"样本 ID {case_id} 重复，出现在第 {', '.join(map(str, lines))} 行",
                lines[-1],
                "id",
            ),
            "case_id": case_id,
            "lines": lines,
        }
        for case_id, lines in id_lines.items()
        if len(lines) > 1
    ]


def _issue(
    code: str, message: str, line: int | None = None, field: str | None = None
) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "message": message}
    if line is not None:
        issue["line"] = line
    if field is not None:
        issue["field"] = field
    return issue


def _raise_if_issues(issues: list[dict[str, Any]]) -> None:
    if issues:
        raise DatasetUploadValidationError(issues)


def _split_multi_value(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]

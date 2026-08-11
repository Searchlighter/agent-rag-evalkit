"""基于样本元数据执行关键词、正则、JSON和引用质量检查。"""

from __future__ import annotations

import json
import re
from typing import Any


def validate_answer(
    answer: str,
    citations: list[str],
    metadata: dict[str, Any],
    retrieval_ids: list[str] | None = None,
) -> dict[str, object]:
    """返回确定性检查结果及可用于诊断的失败明细。"""
    required_keywords = [str(item) for item in metadata.get("required_keywords", [])]
    missing_keywords = [item for item in required_keywords if item.lower() not in answer.lower()]

    missing_patterns: list[str] = []
    invalid_patterns: list[str] = []
    for pattern in [str(item) for item in metadata.get("required_patterns", [])]:
        try:
            if re.search(pattern, answer) is None:
                missing_patterns.append(pattern)
        except re.error:
            invalid_patterns.append(pattern)

    require_citations = bool(metadata.get("require_citations", False))
    known_retrievals = set(retrieval_ids or [])
    invalid_citations = [item for item in citations if item not in known_retrievals]

    schema_errors = _validate_json_schema_subset(answer, metadata.get("answer_json_schema"))
    return {
        "keyword_pass": not missing_keywords,
        "regex_pass": not missing_patterns and not invalid_patterns,
        "json_schema_pass": not schema_errors,
        "citation_presence_pass": not require_citations or bool(citations),
        "citation_validity_pass": not invalid_citations,
        "missing_keywords": missing_keywords,
        "missing_patterns": missing_patterns,
        "invalid_patterns": invalid_patterns,
        "invalid_citations": invalid_citations,
        "schema_errors": schema_errors,
    }


def _validate_json_schema_subset(answer: str, schema: object) -> list[str]:
    """Validate object/type/required/properties from a small JSON Schema subset."""
    if not schema:
        return []
    if not isinstance(schema, dict):
        return ["answer_json_schema must be an object"]
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return ["answer is not valid JSON"]

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_type(payload, str(expected_type)):
        return [f"root must be {expected_type}"]
    if isinstance(payload, dict):
        for name in schema.get("required", []):
            if name not in payload:
                errors.append(f"missing required property: {name}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, rule in properties.items():
                if name in payload and isinstance(rule, dict) and rule.get("type"):
                    if not _matches_type(payload[name], str(rule["type"])):
                        errors.append(f"property {name} must be {rule['type']}")
    return errors


def _matches_type(value: object, expected: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }.get(expected, lambda: False)()

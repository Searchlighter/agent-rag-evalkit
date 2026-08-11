"""将不同 Adapter 事件转换为统一 TraceEvent，并清理常见敏感信息。"""

from __future__ import annotations

import re
from typing import Any

from .domain import TraceEvent, TraceEventType, new_id

_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|token|password)\s*[:=]\s*[^\s,;]+")
_TYPE_ALIASES = {
    "retrieval": "retrieval",
    "retrieve": "retrieval",
    "retriever": "retrieval",
    "retrieval_completed": "retrieval",
    "rerank": "rerank",
    "reranker": "rerank",
    "llm": "llm",
    "model": "llm",
    "tool": "tool",
    "tool_call": "tool",
    "error": "error",
    "final": "final",
    "answer": "final",
    "completed": "final",
}


def redact_text(value: object, max_length: int = 240) -> str:
    """脱敏常见凭据键并限制写入 Trace 的文本长度。"""
    text = str(value or "")
    text = _SECRET_PATTERN.sub(r"\1=[REDACTED]", text)
    return text[:max_length]


def normalize_trace_events(
    raw_events: list[dict[str, Any]],
    *,
    eval_run_id: str,
    case_id: str,
) -> list[TraceEvent]:
    """将 LangGraph Callback 或任意 Adapter 的字典事件映射为统一 Trace。"""
    normalized: list[TraceEvent] = []
    for sequence, raw in enumerate(raw_events, start=1):
        raw_type = str(raw.get("type") or raw.get("event") or "").lower()
        mapped_type = _TYPE_ALIASES.get(raw_type)
        if mapped_type is None:
            mapped_type = "error" if raw.get("error") else "final"
        event_type = TraceEventType(mapped_type)
        duration = raw.get("duration_ms") or raw.get("latency_ms")
        normalized.append(
            TraceEvent(
                id=new_id("trace"),
                trace_id=redact_text(raw.get("trace_id") or f"{eval_run_id}:{case_id}", 120),
                eval_run_id=eval_run_id,
                case_id=case_id,
                event_type=event_type,
                node_name=redact_text(raw.get("node") or raw.get("name") or event_type.value, 80),
                sequence=sequence,
                source=redact_text(raw.get("source") or "adapter", 60),
                source_event_type=redact_text(raw_type, 80),
                duration_ms=float(duration) if duration is not None else None,
                input_summary=redact_text(raw.get("input") or raw.get("query")),
                output_summary=redact_text(raw.get("output") or raw.get("summary")),
                error_message=redact_text(raw.get("error")) or None,
                metadata={
                    key: redact_text(value)
                    for key, value in raw.items()
                    if key not in {"input", "query", "output", "summary", "error", "trace_id", "source"}
                },
            )
        )
    return normalized

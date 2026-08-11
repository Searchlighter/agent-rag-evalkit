"""将标准 TraceEvent 可选导出为 Langfuse span、generation和tool观察项。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from .domain import CaseResult, TraceEvent, TraceEventType


class ObservationProtocol(Protocol):
    """Exporter 使用的最小 Langfuse observation 接口。"""

    def __enter__(self) -> "ObservationProtocol":
        """进入 observation 上下文。"""
        ...

    def __exit__(self, *args: Any) -> bool | None:
        """结束 observation 上下文。"""
        ...

    def update(self, **kwargs: Any) -> Any:
        """更新 observation 输出、级别和用量字段。"""
        ...


class LangfuseClientProtocol(Protocol):
    """避免核心包强依赖 Langfuse SDK 的最小客户端接口。"""

    def start_as_current_observation(self, **kwargs: Any) -> ObservationProtocol:
        """创建当前 observation。"""
        ...

    def flush(self) -> Any:
        """刷新待发送的可观测数据。"""
        ...


@dataclass(slots=True)
class LangfuseTraceExporter:
    """按单条 Case 组织 Langfuse Trace，并关联用量与成本字段。"""
    client: LangfuseClientProtocol
    environment: str = "development"

    def export_case(
        self,
        result: CaseResult,
        events: list[TraceEvent],
        *,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
    ) -> str:
        """导出一次 Case 的完整 Trace，刷新客户端后返回稳定 Trace ID。"""
        trace_id = _langfuse_trace_id(events[0].trace_id if events else f"{result.eval_run_id}:{result.case_id}")
        with self.client.start_as_current_observation(
            as_type="span",
            name="evalkit-case",
            trace_context={"trace_id": trace_id},
            input={"case_id": result.case_id},
            metadata={
                "eval_run_id": result.eval_run_id,
                "case_result_id": result.id,
                "environment": self.environment,
            },
        ) as root:
            for event in events:
                self._export_event(
                    event,
                    model=model,
                    usage_details=usage_details,
                    cost_details=cost_details,
                )
            root.update(
                output={
                    "status": result.status,
                    "answer": result.answer,
                    "citations": result.citations,
                    "metrics": result.metrics,
                }
            )
        self.client.flush()
        return trace_id

    def _export_event(
        self,
        event: TraceEvent,
        *,
        model: str | None,
        usage_details: dict[str, int] | None,
        cost_details: dict[str, float] | None,
    ) -> None:
        observation_type = _observation_type(event.event_type)
        start_kwargs: dict[str, Any] = {
            "as_type": observation_type,
            "name": event.node_name,
            "input": event.input_summary or None,
            "metadata": {
                "evalkit_event_id": event.id,
                "source": event.source,
                "source_event_type": event.source_event_type,
                "sequence": event.sequence,
                "duration_ms": event.duration_ms,
                **event.metadata,
            },
        }
        if observation_type == "generation" and model:
            start_kwargs["model"] = model
        with self.client.start_as_current_observation(**start_kwargs) as observation:
            update_kwargs: dict[str, Any] = {"output": event.output_summary or None}
            if event.error_message:
                update_kwargs.update(level="ERROR", status_message=event.error_message)
            if observation_type == "generation":
                if usage_details:
                    update_kwargs["usage_details"] = usage_details
                if cost_details:
                    update_kwargs["cost_details"] = cost_details
            observation.update(**update_kwargs)


def _langfuse_trace_id(seed: str) -> str:
    """Langfuse custom trace IDs follow W3C's 32-hex-character format."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _observation_type(event_type: TraceEventType) -> str:
    return {
        TraceEventType.RETRIEVAL: "retriever",
        TraceEventType.LLM: "generation",
        TraceEventType.TOOL: "tool",
    }.get(event_type, "span")

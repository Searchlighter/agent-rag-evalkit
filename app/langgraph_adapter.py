"""在不强依赖 LangChain 的前提下采集 LangGraph 回调并适配评测契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .adapter_contract import AdapterResponse, RetrievedChunk


@dataclass(slots=True)
class LangGraphTraceCallback:
    """Callback object accepted by LangGraph/LangChain callback configuration."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _started_at: dict[str, float] = field(default_factory=dict)

    def on_retriever_start(self, serialized: dict[str, Any] | None, query: str, *, run_id: str = "", **_: Any) -> None:
        """记录检索开始事件。"""
        self._start("retrieval", serialized, run_id, query=query)

    def on_retriever_end(self, documents: Any, *, run_id: str = "", **_: Any) -> None:
        """记录检索结果及耗时。"""
        self._end("retrieval", run_id, output=documents)

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: Any, *, run_id: str = "", **_: Any) -> None:
        """记录模型调用开始事件。"""
        self._start("llm", serialized, run_id, input_value=prompts)

    def on_llm_end(self, response: Any, *, run_id: str = "", **_: Any) -> None:
        """记录模型输出及耗时。"""
        self._end("llm", run_id, output=response)

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: str = "", **_: Any) -> None:
        """记录工具调用开始事件。"""
        self._start("tool", serialized, run_id, input_value=input_str)

    def on_tool_end(self, output: Any, *, run_id: str = "", **_: Any) -> None:
        """记录工具输出及耗时。"""
        self._end("tool", run_id, output=output)

    def on_chain_error(self, error: BaseException, *, run_id: str = "", **_: Any) -> None:
        """将图执行异常转换为可标准化错误事件。"""
        self.events.append({"type": "error", "name": "graph", "run_id": str(run_id), "error": str(error)})

    def _start(
        self, event_type: str, serialized: dict[str, Any] | None, run_id: str, **payload: Any
    ) -> None:
        key = str(run_id)
        self._started_at[key] = perf_counter()
        self.events.append({
            "type": event_type,
            "name": _serialized_name(serialized, event_type),
            "run_id": key,
            **payload,
        })

    def _end(self, event_type: str, run_id: str, *, output: Any) -> None:
        key = str(run_id)
        started = self._started_at.pop(key, None)
        event: dict[str, Any] = {
            "type": event_type,
            "name": event_type,
            "run_id": key,
            "output": output,
        }
        if started is not None:
            event["duration_ms"] = round((perf_counter() - started) * 1000, 3)
        self.events.append(event)


@dataclass(slots=True)
class LangGraphCallbackAdapter:
    """Adapt a graph.invoke-compatible callable to EvalKit's TargetAgentAdapter."""

    graph_invoke: Callable[..., dict[str, Any]]

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """执行 graph.invoke，合并回调事件和图状态中的评测字段。"""
        callback = LangGraphTraceCallback()
        state = self.graph_invoke(
            {"question": question},
            config={"callbacks": [callback], "configurable": {"thread_id": request_id}},
        )
        if not isinstance(state, dict) or not isinstance(state.get("answer"), str):
            raise ValueError("LangGraph state must include a string answer")
        retrievals = [_to_retrieved_chunk(item, index) for index, item in enumerate(state.get("retrievals", []), 1)]
        state_events = state.get("events", [])
        if not isinstance(state_events, list):
            raise ValueError("LangGraph state events must be a list")
        return AdapterResponse(
            request_id=request_id,
            answer=state["answer"],
            citations=[str(item) for item in state.get("citations", [])],
            retrievals=retrievals,
            events=[*callback.events, *state_events],
        )


def _serialized_name(serialized: dict[str, Any] | None, fallback: str) -> str:
    if not isinstance(serialized, dict):
        return fallback
    return str(serialized.get("name") or serialized.get("id") or fallback)


def _to_retrieved_chunk(raw: Any, rank: int) -> RetrievedChunk:
    if not isinstance(raw, dict) or not raw.get("document_id") or not raw.get("chunk_id"):
        raise ValueError("LangGraph retrieval requires document_id and chunk_id")
    return RetrievedChunk(
        document_id=str(raw["document_id"]),
        chunk_id=str(raw["chunk_id"]),
        score=float(raw.get("score", 0)),
        rank=int(raw.get("rank", rank)),
        metadata=dict(raw.get("metadata", {})),
        content=str(raw.get("content") or ""),
    )

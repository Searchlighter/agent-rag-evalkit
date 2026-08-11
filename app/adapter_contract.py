"""被测 RAG/Agent 的统一输入输出契约及确定性本地实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class RetrievedChunk:
    """一次检索返回的标准化证据切片。"""

    chunk_id: str
    document_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterResponse:
    """目标系统需要返回给评测引擎的完整单次响应。"""

    request_id: str
    answer: str
    citations: list[str]
    retrievals: list[RetrievedChunk]
    events: list[dict[str, Any]] = field(default_factory=list)


class TargetAgentAdapter(Protocol):
    """所有被测 RAG/Agent 的最小统一接口。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """使用稳定请求ID执行一次问答并返回可评测结果。"""
        ...


class MockRagAdapter:
    """不访问网络的确定性 Adapter，用于快速演示和回归测试。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """返回固定证据和带输入问题的合成回答。"""
        return AdapterResponse(
            request_id=request_id,
            answer=f"模拟回答：{question}",
            citations=["mock-document#chunk-001"],
            retrievals=[
                RetrievedChunk(
                    chunk_id="chunk-001", document_id="mock-document", score=0.91, rank=1
                )
            ],
            events=[{"type": "retrieval", "latency_ms": 8}, {"type": "llm", "latency_ms": 20}],
        )

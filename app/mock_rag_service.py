"""供 HTTP 集成演示使用的确定性合成 RAG 服务。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class SyntheticChunk:
    """内置模拟知识库中的最小文档切片。"""
    document_id: str
    chunk_id: str
    title: str
    content: str
    keywords: tuple[str, ...]

    @property
    def evidence_id(self) -> str:
        """返回与 EvalCase.expected_evidence 一致的复合证据ID。"""
        return f"{self.document_id}#{self.chunk_id}"


SYNTHETIC_CHUNKS = (
    SyntheticChunk(
        "expense-policy",
        "travel",
        "差旅费用管理办法（合成示例）",
        "员工应在出差结束后 10 个工作日内提交报销单，并附发票、行程单和其他有效凭证。",
        ("差旅", "出差", "报销", "发票", "行程", "凭证"),
    ),
    SyntheticChunk(
        "procurement-policy",
        "contract",
        "采购合同审批规范（合成示例）",
        "采购合同先由部门负责人确认业务内容，再依次提交法务合规审查和财务预算审批。",
        ("采购", "合同", "审批", "部门负责人", "法务", "财务"),
    ),
    SyntheticChunk(
        "it-handbook",
        "password",
        "账号与密码手册（合成示例）",
        "忘记密码时，员工可通过统一身份平台完成自助重置；无法验证身份时联系 IT 服务台。",
        ("账号", "密码", "忘记", "重置", "身份平台", "IT"),
    ),
    SyntheticChunk(
        "hr-handbook",
        "annual-leave",
        "员工休假手册（合成示例）",
        "正式员工基础年假为每年 5 个工作日，具体天数根据司龄和当地法规进行调整。",
        ("员工", "年假", "休假", "5 个工作日", "司龄"),
    ),
)


class QueryRequest(BaseModel):
    """模拟 RAG 查询接口的请求结构。"""
    question: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


def build_mock_response(question: str, request_id: str, top_k: int = 2) -> dict[str, Any]:
    """Return contract-compatible output using deterministic keyword overlap."""
    started = perf_counter()
    normalized = question.casefold()
    scored = [
        (sum(keyword.casefold() in normalized for keyword in chunk.keywords), chunk)
        for chunk in SYNTHETIC_CHUNKS
    ]
    matches = [(score, chunk) for score, chunk in scored if score > 0]
    matches.sort(key=lambda item: (-item[0], item[1].evidence_id))
    selected = matches[: max(1, top_k)]
    retrievals = [
        {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "score": round(min(0.99, 0.65 + score * 0.08), 4),
            "rank": rank,
            "content": chunk.content,
            "metadata": {"title": chunk.title, "source": "synthetic"},
        }
        for rank, (score, chunk) in enumerate(selected, start=1)
    ]
    citations = [chunk.evidence_id for _, chunk in selected]
    answer = (
        selected[0][1].content
        if selected
        else "模拟知识库中没有找到相关依据，请补充问题或转人工处理。"
    )
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    return {
        "request_id": request_id,
        "answer": answer,
        "citations": citations,
        "retrievals": retrievals,
        "events": [
            {
                "type": "retrieval",
                "name": "synthetic_keyword_retriever",
                "latency_ms": elapsed_ms,
                "candidate_count": len(matches),
            },
            {"type": "llm", "name": "deterministic_answer_template", "latency_ms": 0},
            {"type": "final", "name": "mock_rag_response"},
        ],
    }


def create_mock_rag_app() -> FastAPI:
    """创建只读取合成知识、不会访问模型或外部网络的 FastAPI 应用。"""
    app = FastAPI(
        title="AgentRAG EvalKit Synthetic RAG",
        version="1.0.0",
        description="Deterministic mock service; synthetic data only.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """报告模拟服务和合成数据状态。"""
        return {"status": "ok", "data": "synthetic"}

    @app.post("/v1/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        """执行关键词匹配并返回 HTTP Adapter 契约结构。"""
        return build_mock_response(request.question, request.request_id)

    return app


app = create_mock_rag_app()

"""从外部语料构建 BM25 与混合检索索引的本地 Reference RAG。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


DEFAULT_CORPUS = Path(__file__).with_name("data") / "reference_corpus.json"
_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")
_QUERY_EXPANSIONS = {
    "差旅费": "报销 发票 行程 凭证",
    "垫付": "报销 发票 费用",
    "协议会签": "合同 审批 法务 财务",
    "登录凭证": "账号 密码 重置",
    "带薪休息": "年假 休假",
    "可疑邮件": "钓鱼邮件 安全团队 上报",
    "诈骗信": "钓鱼邮件 安全团队 上报",
    "笔记本坏了": "设备 损坏 维修 资产服务台",
    "新同事": "新员工 入职 材料",
    "开票": "发票 申请",
    "在家访问内网": "远程办公 VPN 内网",
    "离开公司": "离职 交接",
    "供货商": "供应商 准入",
}


@dataclass(frozen=True, slots=True)
class ReferenceChunk:
    """从外部 JSON 语料加载的一条可引用知识切片。"""

    document_id: str
    chunk_id: str
    title: str
    content: str
    source: str

    @property
    def evidence_id(self) -> str:
        """返回与评测集期望证据一致的复合ID。"""
        return f"{self.document_id}#{self.chunk_id}"


def _tokenize(text: str) -> list[str]:
    """生成英文词及中文一至二元字符，用于无第三方依赖的 BM25。"""
    normalized = text.casefold()
    tokens = _ASCII_WORD.findall(normalized)
    for run in _CHINESE_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _expand_query(query: str) -> str:
    """用显式可审计的领域同义词扩展查询。"""
    additions = [terms for phrase, terms in _QUERY_EXPANSIONS.items() if phrase in query]
    return " ".join([query, *additions])


def _hashed_vector(text: str, dimensions: int = 256) -> dict[int, float]:
    """将字符 n-gram 哈希为稀疏向量，提供可离线复现的语义近似通道。"""
    normalized = re.sub(r"\s+", "", text.casefold())
    grams = [
        normalized[index : index + width]
        for width in (2, 3, 4)
        for index in range(max(0, len(normalized) - width + 1))
    ]
    vector: Counter[int] = Counter()
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest, "big") % dimensions
        vector[slot] += 1
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {slot: value / norm for slot, value in vector.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    """计算两个归一化稀疏向量的余弦相似度。"""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(slot, 0.0) for slot, value in left.items())


class ReferenceRagIndex:
    """在内存中构建 BM25 和哈希向量索引并执行可解释检索。"""

    def __init__(self, chunks: list[ReferenceChunk]) -> None:
        """预计算词频、文档频率、平均长度和稀疏向量。"""
        if not chunks:
            raise ValueError("reference corpus cannot be empty")
        self.chunks = chunks
        self.tokens = [_tokenize(f"{chunk.title} {chunk.content}") for chunk in chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.average_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.document_frequency = Counter(
            term for tokens in self.tokens for term in set(tokens)
        )
        self.vectors = [
            _hashed_vector(_expand_query(f"{chunk.title} {chunk.content}"))
            for chunk in chunks
        ]

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CORPUS) -> "ReferenceRagIndex":
        """从 UTF-8 JSON 文件加载语料并校验必填字段。"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("reference corpus must be a JSON array")
        chunks = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each corpus item must be an object")
            required = ("document_id", "chunk_id", "title", "content", "source")
            if any(not str(item.get(field, "")).strip() for field in required):
                raise ValueError(f"corpus item requires fields: {', '.join(required)}")
            chunks.append(ReferenceChunk(**{field: str(item[field]) for field in required}))
        return cls(chunks)

    def _bm25_scores(self, query: str) -> list[float]:
        """按照 BM25 公式计算查询与全部知识切片的相关性。"""
        query_terms = _tokenize(query)
        scores: list[float] = []
        document_count = len(self.chunks)
        for tokens, frequencies in zip(self.tokens, self.term_frequencies, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                doc_frequency = self.document_frequency[term]
                idf = math.log(1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / self.average_length
                )
                score += idf * frequency * 2.5 / denominator
            scores.append(score)
        return scores

    def search(self, query: str, strategy: str = "hybrid", top_k: int = 3) -> list[dict[str, Any]]:
        """按 BM25 或混合策略返回排序结果与分通道得分。"""
        if strategy not in {"bm25", "hybrid"}:
            raise ValueError("strategy must be bm25 or hybrid")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        expanded_query = _expand_query(query) if strategy == "hybrid" else query
        bm25_scores = self._bm25_scores(expanded_query)
        vector = _hashed_vector(expanded_query)
        vector_scores = [_cosine(vector, item) for item in self.vectors]
        max_bm25 = max(bm25_scores, default=0.0) or 1.0
        ranked = []
        for chunk, bm25, semantic in zip(
            self.chunks, bm25_scores, vector_scores, strict=True
        ):
            normalized_bm25 = bm25 / max_bm25
            combined = normalized_bm25 if strategy == "bm25" else 0.65 * normalized_bm25 + 0.35 * semantic
            ranked.append((combined, bm25, semantic, chunk))
        ranked.sort(key=lambda item: (-item[0], item[3].evidence_id))
        return [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "evidence_id": chunk.evidence_id,
                "title": chunk.title,
                "content": chunk.content,
                "source": chunk.source,
                "score": round(combined, 6),
                "bm25_score": round(bm25, 6),
                "vector_score": round(semantic, 6),
                "rank": rank,
            }
            for rank, (combined, bm25, semantic, chunk) in enumerate(ranked[:top_k], start=1)
            if combined > 0
        ]


class QueryRequest(BaseModel):
    """Reference RAG HTTP 查询参数。"""

    question: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


def build_reference_response(
    index: ReferenceRagIndex,
    question: str,
    request_id: str,
    strategy: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """执行真实索引检索并转换为 EvalKit HTTP Adapter 契约。"""
    started = perf_counter()
    selected = index.search(question, strategy=strategy, top_k=top_k)
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    answer = selected[0]["content"] if selected else "知识库中没有找到足够相关的依据。"
    return {
        "request_id": request_id,
        "answer": answer,
        "citations": [item["evidence_id"] for item in selected[:1]],
        "retrievals": [
            {
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "rank": item["rank"],
                "content": item["content"],
                "metadata": {
                    "title": item["title"],
                    "source": item["source"],
                    "bm25_score": item["bm25_score"],
                    "vector_score": item["vector_score"],
                    "strategy": strategy,
                },
            }
            for item in selected
        ],
        "events": [
            {
                "type": "retrieval",
                "name": f"reference_{strategy}_retriever",
                "latency_ms": elapsed_ms,
                "candidate_count": len(index.chunks),
            },
            {"type": "final", "name": "extractive_reference_answer"},
        ],
    }


def create_reference_rag_app(corpus_path: str | Path = DEFAULT_CORPUS) -> FastAPI:
    """创建从外部语料构建索引的 Reference RAG HTTP 服务。"""
    index = ReferenceRagIndex.from_json(corpus_path)
    app = FastAPI(
        title="AgentRAG EvalKit Reference RAG",
        version="1.0.0",
        description="Local BM25 and hybrid retrieval reference implementation.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """返回索引状态和语料切片数量。"""
        return {"status": "ok", "corpus": "synthetic-public", "chunk_count": len(index.chunks)}

    @app.post("/v1/query/bm25")
    def query_bm25(request: QueryRequest) -> dict[str, Any]:
        """使用 BM25 基线执行检索。"""
        return build_reference_response(index, request.question, request.request_id, "bm25")

    @app.post("/v1/query/hybrid")
    def query_hybrid(request: QueryRequest) -> dict[str, Any]:
        """使用查询扩展、BM25 与哈希向量执行混合检索。"""
        return build_reference_response(index, request.question, request.request_id, "hybrid")

    return app


app = create_reference_rag_app()

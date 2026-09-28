"""将 RAGFlow OpenAI兼容聊天响应映射为 EvalKit AdapterResponse。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .adapter_contract import AdapterResponse, RetrievedChunk


class RagFlowAdapterError(RuntimeError):
    """RAGFlow 传输失败或引用数据不符合契约。"""


@dataclass(slots=True)
class RagFlowChatAdapter:
    """负责 RAGFlow 鉴权、重试及 reference.chunks 字段转换。"""
    base_url: str
    api_key: str
    chat_id: str
    model: str = "model"
    timeout_seconds: float = 30.0
    retries: int = 1

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """调用非流式聊天接口并返回统一评测响应。"""
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not self.api_key.strip() or not self.chat_id.strip():
            raise ValueError("RAGFlow api_key and chat_id are required")
        if self.timeout_seconds <= 0 or self.retries < 0:
            raise ValueError("invalid timeout or retry configuration")

        endpoint = (
            f"{self.base_url.rstrip('/')}/api/v1/openai/"
            f"{quote(self.chat_id, safe='')}/chat/completions"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "extra_body": {
                "reference": True,
                "reference_metadata": {"include": True},
            },
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return self._parse_response(raw, request_id)
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == self.retries:
                    raise RagFlowAdapterError(
                        f"RAGFlow request failed after {attempt + 1} attempt(s): {error}"
                    ) from error
            except json.JSONDecodeError as error:
                raise RagFlowAdapterError("RAGFlow returned invalid JSON") from error
        raise RagFlowAdapterError("unreachable retry state")

    @staticmethod
    def _parse_response(raw: object, request_id: str) -> AdapterResponse:
        if not isinstance(raw, dict):
            raise RagFlowAdapterError("RAGFlow response must be an object")
        choices = raw.get("choices")
        if choices is None or choices == []:
            return AdapterResponse(
                request_id=request_id,
                answer="",
                citations=[],
                retrievals=[],
                events=[],
                missing_fields=["answer", "citations", "retrievals", "trace"],
            )
        if not isinstance(choices, list):
            raise RagFlowAdapterError("RAGFlow response choices must be a list")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RagFlowAdapterError("RAGFlow choice requires message.content")

        missing_fields: list[str] = []
        if "reference" not in message:
            missing_fields.extend(["citations", "retrievals"])
        reference = message.get("reference") or {}
        chunks = reference.get("chunks", {}) if isinstance(reference, dict) else {}
        if isinstance(chunks, dict):
            chunk_rows = list(chunks.values())
        elif isinstance(chunks, list):
            chunk_rows = chunks
        else:
            raise RagFlowAdapterError("RAGFlow reference.chunks must be an object or list")

        retrievals: list[RetrievedChunk] = []
        citations: list[str] = []
        for rank, chunk in enumerate(chunk_rows, start=1):
            if not isinstance(chunk, dict) or not chunk.get("document_id") or not chunk.get("id"):
                raise RagFlowAdapterError("RAGFlow chunk requires document_id and id")
            evidence_id = f"{chunk['document_id']}#{chunk['id']}"
            citations.append(evidence_id)
            retrievals.append(
                RetrievedChunk(
                    document_id=str(chunk["document_id"]),
                    chunk_id=str(chunk["id"]),
                    score=float(chunk.get("similarity", 0)),
                    rank=rank,
                    metadata={
                        key: value
                        for key, value in chunk.items()
                        if key not in {"document_id", "id", "similarity"}
                    },
                )
            )
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return AdapterResponse(
            request_id=request_id,
            answer=message["content"],
            citations=citations,
            retrievals=retrievals,
            events=[
                {
                    "type": "retrieval",
                    "name": "ragflow-reference",
                    "source": "ragflow",
                    "output": {"chunk_count": len(retrievals)},
                },
                {
                    "type": "llm",
                    "name": "ragflow-generation",
                    "source": "ragflow",
                    "output": message["content"],
                    "model": raw.get("model"),
                    "usage": usage,
                },
            ],
            missing_fields=missing_fields,
        )

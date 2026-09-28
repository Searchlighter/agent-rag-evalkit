"""将 Dify Chat App blocking 响应映射为 EvalKit AdapterResponse。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapter_contract import AdapterResponse, RetrievedChunk


class DifyAdapterError(RuntimeError):
    """Dify 传输失败或响应不符合契约。"""


@dataclass(slots=True)
class DifyChatAdapter:
    """负责 Dify 鉴权、请求重试及检索资源字段转换。"""
    base_url: str
    api_key: str
    user: str = "evalkit-demo"
    conversation_id: str = ""
    timeout_seconds: float = 30.0
    retries: int = 1

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """调用 Dify blocking 接口并返回统一评测响应。"""
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not self.api_key.strip():
            raise ValueError("Dify API key is required")
        if self.timeout_seconds <= 0 or self.retries < 0:
            raise ValueError("invalid timeout or retry configuration")

        endpoint = f"{self.base_url.rstrip('/')}/chat-messages"
        payload: dict[str, Any] = {
            "inputs": {},
            "query": question,
            "response_mode": "blocking",
            "user": self.user,
        }
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
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
                    raise DifyAdapterError(
                        f"Dify request failed after {attempt + 1} attempt(s): {error}"
                    ) from error
            except json.JSONDecodeError as error:
                raise DifyAdapterError("Dify returned invalid JSON") from error
        raise DifyAdapterError("unreachable retry state")

    @staticmethod
    def _parse_response(raw: object, request_id: str) -> AdapterResponse:
        if not isinstance(raw, dict):
            raise DifyAdapterError("Dify response must be an object")
        if "answer" in raw and not isinstance(raw["answer"], str):
            raise DifyAdapterError("Dify response must contain a string answer")
        missing_fields: list[str] = []
        answer = raw.get("answer", "")
        if "answer" not in raw:
            missing_fields.append("answer")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise DifyAdapterError("Dify response metadata must be an object")
        if "retriever_resources" not in metadata:
            missing_fields.extend(["citations", "retrievals"])
        resources = metadata.get("retriever_resources") or []
        if not isinstance(resources, list):
            raise DifyAdapterError("metadata.retriever_resources must be a list")

        retrievals: list[RetrievedChunk] = []
        citations: list[str] = []
        for rank, resource in enumerate(resources, start=1):
            if not isinstance(resource, dict):
                raise DifyAdapterError("each Dify retriever resource must be an object")
            document_id = resource.get("document_id")
            chunk_id = resource.get("segment_id") or resource.get("id")
            if not document_id or not chunk_id:
                raise DifyAdapterError(
                    "Dify retriever resource requires document_id and segment_id"
                )
            evidence_id = f"{document_id}#{chunk_id}"
            citations.append(evidence_id)
            retrievals.append(
                RetrievedChunk(
                    document_id=str(document_id),
                    chunk_id=str(chunk_id),
                    score=float(resource.get("score", 0)),
                    rank=int(resource.get("position") or rank),
                    metadata={
                        key: value
                        for key, value in resource.items()
                        if key not in {"document_id", "segment_id", "id", "score", "position"}
                    },
                )
            )
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        return AdapterResponse(
            request_id=request_id,
            answer=answer,
            citations=citations,
            retrievals=retrievals,
            events=[
                {
                    "type": "retrieval",
                    "name": "dify-retriever",
                    "source": "dify",
                    "output": {"resource_count": len(retrievals)},
                },
                {
                    "type": "llm",
                    "name": "dify-generation",
                    "source": "dify",
                    "output": answer,
                    "message_id": raw.get("message_id"),
                    "conversation_id": raw.get("conversation_id"),
                    "usage": usage,
                },
            ],
            missing_fields=missing_fields,
        )

"""通过稳定 JSON 契约接入任意 HTTP RAG/Agent 服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapter_contract import AdapterResponse, RetrievedChunk


class AdapterTransportError(RuntimeError):
    """The target service was unavailable after retry attempts."""


class AdapterContractError(ValueError):
    """The target service returned a payload incompatible with EvalKit."""


@dataclass(slots=True)
class HttpTargetAgentAdapter:
    """封装请求ID、可选鉴权、超时、重试和响应校验。"""
    endpoint: str
    timeout_seconds: float = 15.0
    retries: int = 1
    bearer_token: str | None = None

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """向目标端点发送问题并解析标准响应。"""
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must use http or https")
        if self.timeout_seconds <= 0 or self.retries < 0:
            raise ValueError("timeout_seconds must be positive and retries cannot be negative")

        payload = json.dumps({"question": question, "request_id": request_id}).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return self._parse_response(json.loads(response.read().decode("utf-8")), request_id)
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == self.retries:
                    raise AdapterTransportError(f"HTTP adapter failed after {attempt + 1} attempt(s): {error}") from error
            except json.JSONDecodeError as error:
                raise AdapterContractError("target returned invalid JSON") from error
        raise AdapterTransportError("unreachable adapter retry state")

    @staticmethod
    def _parse_response(raw: dict[str, Any], request_id: str) -> AdapterResponse:
        if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str):
            raise AdapterContractError("response.answer must be a string")
        citations = raw.get("citations", [])
        retrievals = raw.get("retrievals", [])
        events = raw.get("events", [])
        if not all(isinstance(item, str) for item in citations):
            raise AdapterContractError("response.citations must be a string list")
        if not isinstance(retrievals, list) or not isinstance(events, list):
            raise AdapterContractError("response.retrievals and response.events must be lists")

        parsed_retrievals: list[RetrievedChunk] = []
        for index, item in enumerate(retrievals, start=1):
            if not isinstance(item, dict) or not item.get("document_id") or not item.get("chunk_id"):
                raise AdapterContractError("each retrieval requires document_id and chunk_id")
            parsed_retrievals.append(
                RetrievedChunk(
                    document_id=str(item["document_id"]),
                    chunk_id=str(item["chunk_id"]),
                    score=float(item.get("score", 0)),
                    rank=int(item.get("rank", index)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        if not all(isinstance(item, dict) for item in events):
            raise AdapterContractError("response.events must be an object list")
        return AdapterResponse(
            request_id=str(raw.get("request_id") or request_id),
            answer=raw["answer"],
            citations=citations,
            retrievals=parsed_retrievals,
            events=events,
        )

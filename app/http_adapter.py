"""通过稳定 JSON 契约接入任意 HTTP RAG/Agent 服务。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
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

    def __post_init__(self) -> None:
        """在发起网络请求前校验地址、资源限制和鉴权值。"""
        self.endpoint = self.endpoint.strip()
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an absolute http or https URL")
        if parsed.username or parsed.password:
            raise ValueError("endpoint must not contain embedded credentials")
        if parsed.fragment:
            raise ValueError("endpoint must not contain a URL fragment")
        try:
            parsed.port
        except ValueError as error:
            raise ValueError("endpoint contains an invalid port") from error
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 0 and 300")
        if isinstance(self.retries, bool) or not 0 <= self.retries <= 5:
            raise ValueError("retries must be between 0 and 5")
        if self.bearer_token is not None:
            token = self.bearer_token.strip()
            if not token:
                raise ValueError("bearer_token must not be empty")
            self._validate_header_value(token, "bearer_token")
            self.bearer_token = token

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        """向目标端点发送问题并解析标准响应。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        self._validate_header_value(request_id, "request_id")

        payload = json.dumps(
            {"question": question, "request_id": request_id}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw_response = json.loads(response.read().decode("utf-8"))
                    return self._parse_response(raw_response, request_id)
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == self.retries:
                    raise AdapterTransportError(
                        f"HTTP adapter failed after {attempt + 1} attempt(s): {error}"
                    ) from error
            except json.JSONDecodeError as error:
                raise AdapterContractError("target returned invalid JSON") from error
        raise AdapterTransportError("unreachable adapter retry state")

    @staticmethod
    def _validate_header_value(value: str, field_name: str) -> None:
        has_control_character = any(
            ord(character) < 32 or ord(character) == 127 for character in value
        )
        if len(value) > 512 or has_control_character:
            raise ValueError(f"{field_name} contains invalid header characters")

    @staticmethod
    def _parse_response(raw: dict[str, Any], request_id: str) -> AdapterResponse:
        if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str):
            raise AdapterContractError("response.answer must be a string")
        response_request_id = raw.get("request_id", request_id)
        if not isinstance(response_request_id, str) or response_request_id != request_id:
            raise AdapterContractError("response.request_id must match the request")
        citations = raw.get("citations", [])
        retrievals = raw.get("retrievals", [])
        events = raw.get("events", [])
        if not isinstance(citations, list) or not all(
            isinstance(item, str) and item.strip() for item in citations
        ):
            raise AdapterContractError("response.citations must be a non-empty string list")
        if not isinstance(retrievals, list) or not isinstance(events, list):
            raise AdapterContractError("response.retrievals and response.events must be lists")

        parsed_retrievals: list[RetrievedChunk] = []
        for index, item in enumerate(retrievals, start=1):
            if not isinstance(item, dict):
                raise AdapterContractError("each retrieval must be an object")
            document_id = item.get("document_id")
            chunk_id = item.get("chunk_id")
            metadata = item.get("metadata", {})
            if not isinstance(document_id, str) or not document_id.strip():
                raise AdapterContractError("each retrieval requires a string document_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise AdapterContractError("each retrieval requires document_id and chunk_id")
            if not isinstance(metadata, dict):
                raise AdapterContractError("retrieval.metadata must be an object")
            try:
                score = float(item.get("score", 0))
                rank = int(item.get("rank", index))
            except (TypeError, ValueError, OverflowError) as error:
                raise AdapterContractError("retrieval score and rank must be numeric") from error
            if not math.isfinite(score):
                raise AdapterContractError("retrieval.score must be finite")
            if isinstance(item.get("rank", index), bool) or rank < 1:
                raise AdapterContractError("retrieval.rank must be a positive integer")
            parsed_retrievals.append(
                RetrievedChunk(
                    document_id=document_id.strip(),
                    chunk_id=chunk_id.strip(),
                    score=score,
                    rank=rank,
                    metadata=dict(metadata),
                )
            )
        if not all(isinstance(item, dict) for item in events):
            raise AdapterContractError("response.events must be an object list")
        return AdapterResponse(
            request_id=response_request_id,
            answer=raw["answer"],
            citations=citations,
            retrievals=parsed_retrievals,
            events=events,
        )

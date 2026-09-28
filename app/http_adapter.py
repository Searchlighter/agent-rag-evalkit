"""通过稳定 JSON 契约接入任意 HTTP RAG/Agent 服务。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import environ
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .adapter_contract import AdapterResponse, RetrievedChunk


class AdapterTransportError(RuntimeError):
    """The target service was unavailable after retry attempts."""

    code = "adapter_transport_error"

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class AdapterTimeoutError(AdapterTransportError):
    """目标系统在配置的超时时间内没有响应。"""

    code = "adapter_timeout"


class AdapterNetworkError(AdapterTransportError):
    """DNS、连接或其他网络传输失败。"""

    code = "adapter_network_error"


class AdapterHttpStatusError(AdapterTransportError):
    """目标系统返回了非成功 HTTP 状态码。"""

    code = "adapter_http_status"

    def __init__(self, status_code: int, attempts: int) -> None:
        super().__init__(f"target returned HTTP {status_code}", attempts)
        self.status_code = status_code


class AdapterCircuitOpenError(AdapterTransportError):
    """连续失败达到阈值后，熔断器暂时拒绝外部调用。"""

    code = "adapter_circuit_open"

    def __init__(self) -> None:
        super().__init__("adapter circuit is open", attempts=0)


class AdapterContractError(ValueError):
    """The target service returned a payload incompatible with EvalKit."""


@dataclass(slots=True)
class HttpTargetAgentAdapter:
    """封装请求ID、可选鉴权、超时、重试和响应校验。"""

    endpoint: str
    timeout_seconds: float = 15.0
    retries: int = 1
    bearer_token: str | None = field(default=None, repr=False)
    bearer_token_env: str | None = None
    backoff_base_seconds: float = 0.25
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _circuit_opened_at: float | None = field(default=None, init=False, repr=False)

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
        if (
            not math.isfinite(self.backoff_base_seconds)
            or not 0 <= self.backoff_base_seconds <= 10
        ):
            raise ValueError("backoff_base_seconds must be between 0 and 10")
        if (
            isinstance(self.circuit_failure_threshold, bool)
            or not 1 <= self.circuit_failure_threshold <= 100
        ):
            raise ValueError("circuit_failure_threshold must be between 1 and 100")
        if (
            not math.isfinite(self.circuit_recovery_seconds)
            or not 0 < self.circuit_recovery_seconds <= 3600
        ):
            raise ValueError("circuit_recovery_seconds must be between 0 and 3600")
        if self.bearer_token is not None and self.bearer_token_env is not None:
            raise ValueError("configure bearer_token or bearer_token_env, not both")
        if self.bearer_token_env is not None:
            env_name = self.bearer_token_env.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
                raise ValueError("bearer_token_env must be a valid environment variable name")
            token_from_env = environ.get(env_name)
            if token_from_env is None:
                raise ValueError(f"bearer token environment variable is not set: {env_name}")
            self.bearer_token_env = env_name
            self.bearer_token = token_from_env
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
        self._ensure_circuit_available()

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
                    parsed_response = self._parse_response(raw_response, request_id)
                    self._reset_circuit()
                    return parsed_response
            except HTTPError as error:
                retryable = error.code in {408, 425, 429} or error.code >= 500
                classified = AdapterHttpStatusError(error.code, attempt + 1)
                if self._retry_or_raise(classified, error, attempt, retryable):
                    continue
            except TimeoutError as error:
                classified = AdapterTimeoutError(
                    "target request timed out", attempts=attempt + 1
                )
                if self._retry_or_raise(classified, error, attempt, retryable=True):
                    continue
            except URLError as error:
                if isinstance(error.reason, TimeoutError):
                    classified = AdapterTimeoutError(
                        "target request timed out", attempts=attempt + 1
                    )
                else:
                    classified = AdapterNetworkError(
                        "target network request failed", attempts=attempt + 1
                    )
                if self._retry_or_raise(classified, error, attempt, retryable=True):
                    continue
            except json.JSONDecodeError as error:
                raise AdapterContractError("target returned invalid JSON") from error
        raise RuntimeError("unreachable adapter retry state")

    def _retry_or_raise(
        self,
        classified: AdapterTransportError,
        original: Exception,
        attempt: int,
        retryable: bool,
    ) -> bool:
        """对可恢复错误退避重试；最终失败计入熔断器。"""
        if retryable and attempt < self.retries:
            sleep(self.backoff_base_seconds * (2**attempt))
            return True
        self._record_failure()
        raise classified from original

    def _ensure_circuit_available(self) -> None:
        if self._circuit_opened_at is None:
            return
        if monotonic() - self._circuit_opened_at < self.circuit_recovery_seconds:
            raise AdapterCircuitOpenError()
        self._reset_circuit()

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_failure_threshold:
            self._circuit_opened_at = monotonic()

    def _reset_circuit(self) -> None:
        self._consecutive_failures = 0
        self._circuit_opened_at = None

    @staticmethod
    def _validate_header_value(value: str, field_name: str) -> None:
        has_control_character = any(
            ord(character) < 32 or ord(character) == 127 for character in value
        )
        if len(value) > 512 or has_control_character:
            raise ValueError(f"{field_name} contains invalid header characters")

    @staticmethod
    def _parse_response(raw: dict[str, Any], request_id: str) -> AdapterResponse:
        if not isinstance(raw, dict):
            raise AdapterContractError("response must be an object")
        if "answer" in raw and not isinstance(raw["answer"], str):
            raise AdapterContractError("response.answer must be a string")
        missing_fields = [
            logical_name
            for payload_name, logical_name in (
                ("answer", "answer"),
                ("citations", "citations"),
                ("retrievals", "retrievals"),
                ("events", "trace"),
            )
            if payload_name not in raw
        ]
        response_request_id = raw.get("request_id", request_id)
        if not isinstance(response_request_id, str) or response_request_id != request_id:
            raise AdapterContractError("response.request_id must match the request")
        answer = raw.get("answer", "")
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
            answer=answer,
            citations=citations,
            retrievals=parsed_retrievals,
            events=events,
            missing_fields=missing_fields,
        )

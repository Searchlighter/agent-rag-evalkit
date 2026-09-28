"""通用 HTTP Adapter 配置、请求头和响应契约测试。"""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from app.http_adapter import (
    AdapterCircuitOpenError,
    AdapterContractError,
    AdapterHttpStatusError,
    AdapterNetworkError,
    AdapterTimeoutError,
    HttpTargetAgentAdapter,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class HttpAdapterValidationTests(unittest.TestCase):
    def test_transient_http_errors_use_exponential_backoff(self) -> None:
        adapter = HttpTargetAgentAdapter(
            "https://example.test/query",
            retries=2,
            backoff_base_seconds=0.5,
        )
        unavailable = HTTPError(
            adapter.endpoint, 503, "unavailable", hdrs=None, fp=None
        )

        with (
            patch("app.http_adapter.urlopen", side_effect=unavailable) as mocked,
            patch("app.http_adapter.sleep") as mocked_sleep,
            self.assertRaises(AdapterHttpStatusError) as context,
        ):
            adapter.invoke("question", request_id="request-1")

        self.assertEqual(3, mocked.call_count)
        self.assertEqual([0.5, 1.0], [call.args[0] for call in mocked_sleep.call_args_list])
        self.assertEqual(503, context.exception.status_code)
        self.assertEqual(3, context.exception.attempts)

    def test_non_retryable_http_error_fails_immediately(self) -> None:
        adapter = HttpTargetAgentAdapter(
            "https://example.test/query", retries=3, backoff_base_seconds=0
        )
        bad_request = HTTPError(adapter.endpoint, 400, "bad request", None, None)

        with (
            patch("app.http_adapter.urlopen", side_effect=bad_request) as mocked,
            patch("app.http_adapter.sleep") as mocked_sleep,
            self.assertRaises(AdapterHttpStatusError) as context,
        ):
            adapter.invoke("question", request_id="request-1")

        self.assertEqual(1, mocked.call_count)
        mocked_sleep.assert_not_called()
        self.assertEqual(400, context.exception.status_code)

    def test_timeout_and_network_failures_have_distinct_error_codes(self) -> None:
        timeout_adapter = HttpTargetAgentAdapter(
            "https://example.test/query", retries=0
        )
        with (
            patch("app.http_adapter.urlopen", side_effect=URLError(TimeoutError())),
            self.assertRaises(AdapterTimeoutError) as timeout_context,
        ):
            timeout_adapter.invoke("question", request_id="request-1")

        network_adapter = HttpTargetAgentAdapter(
            "https://example.test/query", retries=0
        )
        with (
            patch("app.http_adapter.urlopen", side_effect=URLError("dns failure")),
            self.assertRaises(AdapterNetworkError) as network_context,
        ):
            network_adapter.invoke("question", request_id="request-2")

        self.assertEqual("adapter_timeout", timeout_context.exception.code)
        self.assertEqual("adapter_network_error", network_context.exception.code)

    def test_circuit_opens_after_threshold_and_recovers_after_window(self) -> None:
        adapter = HttpTargetAgentAdapter(
            "https://example.test/query",
            retries=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=30,
        )
        with patch("app.http_adapter.urlopen", side_effect=URLError("offline")) as mocked:
            for request_id in ("request-1", "request-2"):
                with self.assertRaises(AdapterNetworkError):
                    adapter.invoke("question", request_id=request_id)
            with self.assertRaises(AdapterCircuitOpenError):
                adapter.invoke("question", request_id="request-3")
        self.assertEqual(2, mocked.call_count)

        payload = {
            "request_id": "request-4",
            "answer": "answer",
            "citations": [],
            "retrievals": [],
            "events": [],
        }
        assert adapter._circuit_opened_at is not None
        recovered_at = adapter._circuit_opened_at + 31
        with (
            patch("app.http_adapter.monotonic", return_value=recovered_at),
            patch("app.http_adapter.urlopen", return_value=FakeResponse(payload)),
        ):
            result = adapter.invoke("question", request_id="request-4")

        self.assertEqual("answer", result.answer)
        self.assertEqual(0, adapter._consecutive_failures)
        self.assertIsNone(adapter._circuit_opened_at)

    def test_missing_response_fields_are_preserved_for_evaluation_diagnosis(self) -> None:
        response = HttpTargetAgentAdapter._parse_response(
            {"request_id": "request-1"}, "request-1"
        )

        self.assertEqual("", response.answer)
        self.assertEqual([], response.citations)
        self.assertEqual([], response.retrievals)
        self.assertEqual(["answer", "citations", "retrievals", "trace"], response.missing_fields)

    def test_configuration_rejects_invalid_urls_limits_and_header_injection(self) -> None:
        invalid_options = [
            {"endpoint": "ftp://example.test/query"},
            {"endpoint": "https:///missing-host"},
            {"endpoint": "https://user:password@example.test/query"},
            {"endpoint": "https://example.test/query#fragment"},
            {"endpoint": "https://example.test/query", "timeout_seconds": 0},
            {"endpoint": "https://example.test/query", "timeout_seconds": 301},
            {"endpoint": "https://example.test/query", "retries": 6},
            {
                "endpoint": "https://example.test/query",
                "bearer_token": "token\r\nX-Injected: true",
            },
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                HttpTargetAgentAdapter(**options)

    def test_invoke_sends_validated_auth_and_request_id_headers(self) -> None:
        payload = {
            "request_id": "request-1",
            "answer": "answer",
            "citations": ["doc-a#chunk-1"],
            "retrievals": [
                {"document_id": "doc-a", "chunk_id": "chunk-1", "score": 0.9}
            ],
            "events": [],
        }
        adapter = HttpTargetAgentAdapter(
            "https://example.test/query", bearer_token=" test-token ", retries=0
        )

        with patch("app.http_adapter.urlopen", return_value=FakeResponse(payload)) as mocked:
            result = adapter.invoke("question", request_id="request-1")

        request = mocked.call_args.args[0]
        self.assertEqual("Bearer test-token", request.get_header("Authorization"))
        self.assertEqual("request-1", request.get_header("X-request-id"))
        self.assertEqual("request-1", result.request_id)
        with self.assertRaises(ValueError):
            adapter.invoke("question", request_id="request\r\nX-Injected: true")

    def test_response_contract_rejects_ambiguous_or_invalid_fields(self) -> None:
        valid = {
            "request_id": "request-1",
            "answer": "answer",
            "citations": ["doc-a#chunk-1"],
            "retrievals": [
                {
                    "document_id": "doc-a",
                    "chunk_id": "chunk-1",
                    "score": 0.9,
                    "rank": 1,
                    "metadata": {},
                }
            ],
            "events": [],
        }
        invalid_payloads = [
            {**valid, "request_id": "different-request"},
            {**valid, "citations": "doc-a#chunk-1"},
            {**valid, "retrievals": [{"document_id": 1, "chunk_id": "chunk-1"}]},
            {**valid, "retrievals": [{"document_id": "doc-a", "chunk_id": 1}]},
            {
                **valid,
                "retrievals": [
                    {"document_id": "doc-a", "chunk_id": "chunk-1", "score": "nan"}
                ],
            },
            {
                **valid,
                "retrievals": [
                    {"document_id": "doc-a", "chunk_id": "chunk-1", "rank": 0}
                ],
            },
            {
                **valid,
                "retrievals": [
                    {"document_id": "doc-a", "chunk_id": "chunk-1", "metadata": []}
                ],
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(AdapterContractError):
                HttpTargetAgentAdapter._parse_response(payload, "request-1")

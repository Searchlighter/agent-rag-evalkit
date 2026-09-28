"""通用 HTTP Adapter 配置、请求头和响应契约测试。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.http_adapter import AdapterContractError, HttpTargetAgentAdapter


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

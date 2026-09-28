"""Adapter 注册表与 API 解析行为测试。"""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.adapter_contract import AdapterResponse, RetrievedChunk
from app.adapter_registry import AdapterNotFoundError, AdapterRegistry
from app.api import TestAdapterRequest, create_router
from app.domain import AdapterConfig, AdapterType, AdapterVersion
from app.repository import InMemoryRepository
from app.service import EvalKitService


class MatchingAdapter:
    """返回与测试样本匹配证据的确定性 Adapter。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        return AdapterResponse(
            request_id=request_id,
            answer=f"answer: {question}",
            citations=["doc-a#chunk-1"],
            retrievals=[RetrievedChunk("chunk-1", "doc-a", 0.9, 1)],
        )


class FailingAdapter:
    """模拟包含敏感异常文本的连接失败。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        raise RuntimeError("connection failed with token=secret-value")


class AdapterRegistryTests(unittest.TestCase):
    def test_adapter_connection_endpoint_returns_safe_contract_summary(self) -> None:
        service = EvalKitService(InMemoryRepository())
        router = create_router(
            service, AdapterRegistry({"custom-http": MatchingAdapter()})
        )
        probe = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/adapters/test"
        )

        result = probe(TestAdapterRequest(adapter_id="custom-http", question="ping"))

        self.assertEqual("ok", result["status"])
        self.assertTrue(result["response_request_id_matches"])
        self.assertTrue(result["answer_present"])
        self.assertEqual(1, result["citation_count"])
        self.assertEqual(1, result["retrieval_count"])
        self.assertNotIn("answer", result)

        with self.assertRaises(HTTPException) as context:
            probe(TestAdapterRequest(adapter_id="missing"))
        self.assertEqual(404, context.exception.status_code)

    def test_adapter_connection_endpoint_hides_internal_error_text(self) -> None:
        service = EvalKitService(InMemoryRepository())
        router = create_router(service, AdapterRegistry({"failing": FailingAdapter()}))
        probe = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/adapters/test"
        )

        with self.assertRaises(HTTPException) as context:
            probe(TestAdapterRequest(adapter_id="failing"))

        self.assertEqual(502, context.exception.status_code)
        self.assertEqual("adapter_connection_failed", context.exception.detail["code"])
        self.assertNotIn("secret-value", str(context.exception.detail))

    def test_adapter_domain_models_validate_and_redact_version_snapshot(self) -> None:
        config = AdapterConfig("adapter-1", " HR assistant ", AdapterType.HTTP)
        version = AdapterVersion(
            id="adapter-version-1",
            adapter_config_id=config.id,
            version_number=1,
            adapter_type=config.adapter_type,
            settings={"endpoint": "https://example.test/query", "timeout_seconds": 5},
            secret_refs={"bearer_token": "EVALKIT_HTTP_TOKEN"},
        )

        self.assertEqual("HR assistant", config.name)
        self.assertEqual("https://example.test/query", version.settings["endpoint"])
        self.assertEqual("***", version.public_snapshot()["secret_refs"]["bearer_token"])
        with self.assertRaises(TypeError):
            version.settings["endpoint"] = "https://malicious.test"  # type: ignore[index]

    def test_adapter_version_rejects_invalid_version_and_complex_settings(self) -> None:
        with self.assertRaises(ValueError):
            AdapterVersion("v1", "adapter-1", 0, AdapterType.HTTP)
        with self.assertRaises(ValueError):
            AdapterVersion(
                "v1",
                "adapter-1",
                1,
                AdapterType.HTTP,
                settings={"headers": {"X-Test": "value"}},  # type: ignore[dict-item]
            )
        with self.assertRaises(ValueError):
            AdapterVersion(
                "v1",
                "adapter-1",
                1,
                AdapterType.HTTP,
                settings={"api_key": "must-not-be-stored-here"},
            )

    def test_registry_resolves_registered_adapter_and_rejects_unknown_id(self) -> None:
        adapter = MatchingAdapter()
        registry = AdapterRegistry({"custom-http": adapter})

        self.assertIs(adapter, registry.resolve("custom-http"))
        self.assertEqual(("custom-http",), registry.list_ids())
        with self.assertRaises(AdapterNotFoundError):
            registry.resolve("missing")

    def test_registry_rejects_duplicate_registration_without_replace(self) -> None:
        registry = AdapterRegistry({"custom-http": MatchingAdapter()})

        with self.assertRaises(ValueError):
            registry.register("custom-http", MatchingAdapter())

    def test_execute_api_uses_adapter_id_bound_to_run(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("adapter test", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-1",
                "question": "policy",
                "expected_answers": ["answer"],
                "expected_evidence": ["doc-a#chunk-1"],
            }],
            "tester",
        )
        config = service.create_config("top-1", 1)
        run = service.create_eval_run(version.id, config.id, "custom-http")
        router = create_router(
            service, AdapterRegistry({"custom-http": MatchingAdapter()})
        )
        execute = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs/{run_id}/execute"
        )

        summary = execute(run.id)

        self.assertEqual("succeeded", summary["status"])
        self.assertEqual("custom-http", summary["adapter_id"])
        self.assertEqual(1.0, summary["metrics"]["recall_at_k"])

    def test_eval_run_keeps_adapter_version_and_configuration_snapshot(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("snapshot test", "tester")
        version = service.import_dataset_version(
            dataset.id, [{"id": "case-1", "question": "policy"}], "tester"
        )
        config = service.create_config("top-1", 1)
        source_snapshot = {
            "endpoint": "https://example.test/query",
            "timeout_seconds": 5,
        }

        run = service.create_eval_run(
            version.id,
            config.id,
            "custom-http",
            adapter_version_id="adapter-version-1",
            adapter_snapshot=source_snapshot,
            model_id="agent-release-2026-09",
        )
        source_snapshot["endpoint"] = "https://changed.test/query"
        summary = service.get_run_summary(run.id)

        self.assertEqual("adapter-version-1", summary["adapter_version_id"])
        self.assertEqual("agent-release-2026-09", summary["model_id"])
        self.assertEqual(
            "https://example.test/query", summary["adapter_snapshot"]["endpoint"]
        )
        with self.assertRaises(ValueError):
            service.create_eval_run(
                version.id,
                config.id,
                "custom-http",
                adapter_snapshot={"api_token": "must-not-be-stored"},
            )


if __name__ == "__main__":
    unittest.main()

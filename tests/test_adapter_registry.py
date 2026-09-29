"""Adapter 注册表与 API 解析行为测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.adapter_contract import AdapterResponse, RetrievedChunk
from app.adapter_registry import AdapterDisabledError, AdapterNotFoundError, AdapterRegistry
from app.api import (
    ConfigureHttpAdapterRequest,
    CreateAdapterRequest,
    CreateEvalRunRequest,
    TestAdapterRequest,
    UpdateAdapterRequest,
    create_router,
)
from app.domain import AdapterConfig, AdapterType, AdapterVersion
from app.http_adapter import HttpTargetAgentAdapter
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


class SecretAnswerAdapter:
    """模拟目标系统把敏感值放入回答，连接探测仍不得回显正文。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        return AdapterResponse(
            request_id=request_id,
            answer="runtime-secret",
            citations=[],
            retrievals=[],
        )


class IncompleteAdapter:
    """模拟成功返回但缺少评测所需字段的目标系统。"""

    def invoke(self, question: str, *, request_id: str) -> AdapterResponse:
        return AdapterResponse(
            request_id=request_id,
            answer="",
            citations=[],
            retrievals=[],
            missing_fields=["answer", "citations", "retrievals", "trace"],
        )


class AdapterRegistryTests(unittest.TestCase):
    def test_missing_adapter_fields_create_not_evaluable_case(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("incomplete response", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-1",
                "question": "policy",
                "expected_evidence": ["doc-a#chunk-1"],
            }],
            "tester",
        )
        config = service.create_config("top-1", 1)
        run = service.create_eval_run(version.id, config.id, "incomplete")

        summary = service.execute_eval_run(run.id, IncompleteAdapter())
        result = service.list_case_results(run.id)[0]

        self.assertEqual("not_evaluable", result.status)
        self.assertEqual(1, summary["not_evaluable_case_count"])
        self.assertEqual(
            ["missing_answer", "missing_citations", "missing_retrievals", "missing_trace"],
            result.not_evaluable_reasons,
        )
        self.assertTrue(all(value is None for value in result.metrics.values()))

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

    def test_adapter_connection_endpoint_never_echoes_answer_content(self) -> None:
        service = EvalKitService(InMemoryRepository())
        router = create_router(
            service, AdapterRegistry({"secret-answer": SecretAnswerAdapter()})
        )
        probe = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/adapters/test"
        )

        result = probe(TestAdapterRequest(adapter_id="secret-answer"))

        self.assertTrue(result["answer_present"])
        self.assertNotIn("runtime-secret", str(result))

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

    def test_registry_explicit_replace_and_sorted_ids(self) -> None:
        original = MatchingAdapter()
        replacement = IncompleteAdapter()
        registry = AdapterRegistry({"z-adapter": original})
        registry.register("a-adapter", MatchingAdapter())
        registry.register("z-adapter", replacement, replace=True)

        self.assertEqual(("a-adapter", "z-adapter"), registry.list_ids())
        self.assertIs(replacement, registry.resolve("z-adapter"))

    def test_adapter_management_api_creates_edits_toggles_and_deletes(self) -> None:
        registry = AdapterRegistry({"runtime-http": MatchingAdapter()})
        router = create_router(EvalKitService(InMemoryRepository()), registry)

        def endpoint(path: str, method: str):
            return next(
                route.endpoint
                for route in router.routes
                if route.path == path and method in route.methods
            )

        create = endpoint("/api/v1/adapters", "POST")
        list_all = endpoint("/api/v1/adapters", "GET")
        update = endpoint("/api/v1/adapters/{adapter_id}", "PATCH")
        delete = endpoint("/api/v1/adapters/{adapter_id}", "DELETE")

        created = create(
            CreateAdapterRequest(
                adapter_id="team-rag", name="团队知识库", adapter_type="http"
            )
        )
        self.assertFalse(created["configured"])
        self.assertEqual(["runtime-http", "team-rag"], [item["id"] for item in list_all()])

        renamed = update(
            "team-rag", UpdateAdapterRequest(name="团队知识库 V2", enabled=False)
        )
        self.assertEqual("团队知识库 V2", renamed["name"])
        self.assertFalse(renamed["enabled"])

        registry.update_config("runtime-http", enabled=False)
        with self.assertRaises(AdapterDisabledError):
            registry.resolve("runtime-http")

        deleted = delete("team-rag")
        self.assertEqual("deleted", deleted["status"])
        self.assertEqual(["runtime-http"], [item["id"] for item in list_all()])

        with self.assertRaises(HTTPException) as duplicate:
            create(
                CreateAdapterRequest(
                    adapter_id="runtime-http", name="重复配置", adapter_type="http"
                )
            )
        self.assertEqual(409, duplicate.exception.status_code)

    def test_http_adapter_configuration_creates_version_and_supports_probe(self) -> None:
        registry = AdapterRegistry()
        registry.create_config("team-rag", "团队知识库", AdapterType.HTTP)
        router = create_router(EvalKitService(InMemoryRepository()), registry)

        configure = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/adapters/{adapter_id}/http-configuration"
        )
        probe = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/adapters/test"
        )

        configured = configure(
            "team-rag",
            ConfigureHttpAdapterRequest(
                endpoint=" https://example.test/query ",
                auth_method="none",
                timeout_seconds=8,
                retries=2,
            ),
        )

        self.assertTrue(configured["configured"])
        self.assertEqual(1, configured["connection"]["version_number"])
        self.assertEqual(
            "https://example.test/query", configured["connection"]["endpoint"]
        )
        self.assertIsInstance(registry.resolve("team-rag"), HttpTargetAgentAdapter)

        response = AdapterResponse(
            request_id="replaced-by-probe",
            answer="available",
            citations=[],
            retrievals=[],
        )
        with patch.object(HttpTargetAgentAdapter, "invoke", return_value=response):
            result = probe(TestAdapterRequest(adapter_id="team-rag", question="ping"))
        self.assertEqual("ok", result["status"])
        self.assertTrue(result["answer_present"])

    def test_http_adapter_configuration_uses_environment_reference_without_token(self) -> None:
        registry = AdapterRegistry()
        registry.create_config("secure-rag", "安全知识库", AdapterType.HTTP)

        with patch.dict(
            "app.http_adapter.environ", {"TEAM_RAG_TOKEN": "runtime-secret"}
        ):
            configured = registry.configure_http(
                "secure-rag",
                endpoint="https://example.test/query",
                timeout_seconds=15,
                retries=1,
                auth_method="bearer_env",
                bearer_token_env="TEAM_RAG_TOKEN",
            )

        self.assertEqual(
            "TEAM_RAG_TOKEN", configured["connection"]["bearer_token_env"]
        )
        self.assertNotIn("runtime-secret", str(configured))

    def test_create_run_api_uses_current_adapter_version_snapshot(self) -> None:
        repository = InMemoryRepository()
        service = EvalKitService(repository)
        dataset = service.create_dataset("页面任务", "tester")
        version = service.import_dataset_version(
            dataset.id, [{"id": "case-1", "question": "测试问题"}]
        )
        evaluation_config = service.create_config("top-3", 3)
        registry = AdapterRegistry()
        registry.create_config("team-rag", "团队知识库", AdapterType.HTTP)
        configured = registry.configure_http(
            "team-rag",
            endpoint="https://example.test/query",
            timeout_seconds=8,
            retries=1,
            auth_method="none",
        )
        router = create_router(service, registry)
        create_run = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs" and "POST" in route.methods
        )
        list_runs = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs" and "GET" in route.methods
        )
        execute_run = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs/{run_id}/execute"
        )
        list_results = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs/{run_id}/results"
        )

        run = create_run(
            CreateEvalRunRequest(
                dataset_version_id=version.id,
                config_id=evaluation_config.id,
                adapter_id="team-rag",
                model_id="release-1",
            )
        )

        self.assertEqual(configured["current_version_id"], run["adapter_version_id"])
        self.assertEqual(
            "https://example.test/query", run["adapter_snapshot"]["endpoint"]
        )
        self.assertEqual("queued", run["status"].value)

        queued = list_runs()[0]
        self.assertEqual("页面任务", queued["dataset_name"])
        self.assertEqual("user", queued["source"])
        self.assertEqual("queued", queued["status"])
        response = AdapterResponse(
            request_id="runtime-request",
            answer="测试回答",
            citations=["doc-a#chunk-1"],
            retrievals=[
                RetrievedChunk(
                    "chunk-1",
                    "doc-a",
                    0.95,
                    1,
                    {"title": "测试文档"},
                    content="这是召回的原始 Chunk 文本。",
                )
            ],
        )
        with patch.object(HttpTargetAgentAdapter, "invoke", return_value=response):
            executed = execute_run(run["id"])
        self.assertEqual("succeeded", executed["status"])
        self.assertEqual(1, executed["completed_case_count"])
        self.assertEqual("succeeded", list_runs()[0]["status"])
        case_result = list_results(run["id"])[0]
        self.assertEqual("测试问题", case_result["question"])
        self.assertEqual("测试回答", case_result["answer"])
        self.assertEqual(["doc-a#chunk-1"], case_result["citations"])
        self.assertEqual(
            "这是召回的原始 Chunk 文本。",
            case_result["retrieval_candidates"][0]["content"],
        )

        registry.update_config("team-rag", enabled=False)
        with self.assertRaises(HTTPException) as context:
            create_run(
                CreateEvalRunRequest(
                    dataset_version_id=version.id,
                    config_id=evaluation_config.id,
                    adapter_id="team-rag",
                )
            )
        self.assertEqual(409, context.exception.status_code)

    def test_execute_api_rejects_run_bound_to_unknown_adapter(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("unknown adapter", "tester")
        version = service.import_dataset_version(
            dataset.id, [{"id": "case-1", "question": "policy"}], "tester"
        )
        config = service.create_config("top-1", 1)
        run = service.create_eval_run(version.id, config.id, "missing-adapter")
        router = create_router(service, AdapterRegistry())
        execute = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/eval-runs/{run_id}/execute"
        )

        with self.assertRaises(HTTPException) as context:
            execute(run.id)

        self.assertEqual(409, context.exception.status_code)
        self.assertIn("missing-adapter", str(context.exception.detail))

    def test_adapter_failure_is_redacted_in_result_trace_and_csv(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("failed adapter", "tester")
        version = service.import_dataset_version(
            dataset.id, [{"id": "case-1", "question": "policy"}], "tester"
        )
        config = service.create_config("top-1", 1)
        run = service.create_eval_run(version.id, config.id, "failing")

        summary = service.execute_eval_run(run.id, FailingAdapter())
        result = service.list_case_results(run.id)[0]
        trace = service.get_case_trace(run.id, "case-1")
        exported = service.export_case_results_csv(run.id)

        self.assertEqual(1, summary["failed_case_count"])
        self.assertEqual("failed", result.status)
        self.assertIn("token=[REDACTED]", result.error or "")
        self.assertNotIn("secret-value", result.error or "")
        self.assertNotIn("secret-value", exported)
        self.assertNotIn("secret-value", str(trace))

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

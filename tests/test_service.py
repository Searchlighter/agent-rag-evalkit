"""核心评测、诊断、Adapter和回归能力的集成测试。"""

import unittest
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

from app.adapter_contract import AdapterResponse, RetrievedChunk
from app.benchmark import run_benchmark
from app.dify_adapter import DifyChatAdapter
from app.domain import BadcaseStatus, EvalRunStatus
from app.ingestion import load_csv_cases, load_jsonl_cases
from app.http_adapter import AdapterContractError, HttpTargetAgentAdapter
from app.langgraph_adapter import LangGraphCallbackAdapter
from app.langfuse_exporter import LangfuseTraceExporter
from app.mcp_server import EvalKitMcpServer, McpSecurityConfig
from app.main import create_app
from app.repository import InMemoryRepository
from app.quality import validate_answer
from app.ragflow_adapter import RagFlowChatAdapter
from app.service import EvalKitService
from app.trace import normalize_trace_events


class EvalKitServiceTests(unittest.TestCase):
    def test_dataset_version_and_run_state(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("test", "tester")
        version = service.import_dataset_version(dataset.id, [{"id": "c1", "question": "test"}])
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "mock")
        service.update_eval_run_status(run.id, EvalRunStatus.RUNNING)
        completed = service.update_eval_run_status(run.id, EvalRunStatus.SUCCEEDED)
        self.assertEqual(completed.status, EvalRunStatus.SUCCEEDED)

    def test_sync_evaluation_calculates_retrieval_metrics(self) -> None:
        class MatchingAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(
                    request_id=request_id,
                    answer=f"回答：{question}",
                    citations=["doc-a#chunk-1"],
                    retrievals=[
                        RetrievedChunk(
                            chunk_id="chunk-1",
                            document_id="doc-a",
                            score=0.98,
                            rank=1,
                        )
                    ],
                    events=[{"event": "retrieval_completed", "request_id": request_id}],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("企业知识库", "用于测试")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-1",
                "question": "报销政策是什么？",
                "expected_evidence": ["doc-a#chunk-1"],
            }],
        )
        config = service.create_config("baseline", retrieval_k=5)
        run = service.create_eval_run(version.id, config.id, "matching-adapter")

        summary = service.execute_eval_run(run.id, MatchingAdapter())
        results = service.list_case_results(run.id)

        self.assertEqual(EvalRunStatus.SUCCEEDED.value, summary["status"])
        self.assertEqual(1, summary["completed_case_count"])
        self.assertEqual(1.0, summary["metrics"]["recall_at_k"])
        self.assertEqual(1.0, summary["metrics"]["mrr"])
        self.assertEqual(1.0, summary["metrics"]["citation_coverage"])
        self.assertEqual("succeeded", results[0].status)

    def test_jsonl_ingestion_loads_sample_cases(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = load_jsonl_cases(root / "sample_data" / "enterprise_eval_cases.jsonl")
        self.assertGreaterEqual(len(cases), 1)
        self.assertTrue(cases[0]["question"])

    def test_dataset_import_rejects_string_instead_of_string_array(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("strict-schema", "tester")
        with self.assertRaisesRegex(ValueError, "expected_evidence"):
            service.import_dataset_version(
                dataset.id,
                [{"question": "test", "expected_evidence": "doc#chunk"}],
            )

    def test_csv_ingestion_empty_retrieval_and_result_export(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text(
                "id,question,expected_evidence,tags\ncase-csv,发票怎么报销,doc-a#chunk-1|doc-b#chunk-1,finance|policy\n",
                encoding="utf-8",
            )
            cases = load_csv_cases(csv_path)
        self.assertEqual(["doc-a#chunk-1", "doc-b#chunk-1"], cases[0]["expected_evidence"])

        class EmptyAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(request_id=request_id, answer="未找到", citations=[], retrievals=[])

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("csv-test", "tester")
        version = service.import_dataset_version(dataset.id, cases)
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "empty")
        summary = service.execute_eval_run(run.id, EmptyAdapter())
        exported = service.export_case_results_csv(run.id)
        self.assertEqual(1.0, summary["metrics"]["empty_retrieval_rate"])
        self.assertIn("case-csv,succeeded", exported)

    def test_trace_badcase_and_regression_suite(self) -> None:
        class NoRetrievalAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(
                    request_id=request_id,
                    answer="未找到相关信息",
                    citations=[],
                    retrievals=[],
                    events=[{
                        "type": "retrieval",
                        "name": "retrieve",
                        "latency_ms": 12,
                        "input": "api_key=secret-value",
                    }],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("trace-test", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-trace",
                "question": "报销政策是什么？",
                "expected_evidence": ["doc-a#chunk-1"],
                "metadata": {"require_citations": True},
            }],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "no-retrieval")
        service.execute_eval_run(run.id, NoRetrievalAdapter())

        trace = service.get_case_trace(run.id, "case-trace")
        diagnosis = service.diagnose_case(run.id, "case-trace")
        self.assertEqual(1, len(trace))
        self.assertIn("[REDACTED]", trace[0].input_summary)
        self.assertEqual("no_retrieval", diagnosis["suggested_category"])

        badcase = service.create_badcase(run.id, "case-trace", "no_retrieval", "未召回证据")
        updated = service.update_badcase(
            badcase.id, BadcaseStatus.TRIAGED, root_cause="索引缺失", owner="rag-owner"
        )
        suite = service.create_regression_suite_from_badcases("retrieval-regression", [badcase.id])
        comparison = service.compare_regression_runs(suite.id, run.id, run.id)
        self.assertEqual(BadcaseStatus.TRIAGED, updated.status)
        self.assertEqual("needs_manual_review", comparison["comparison"]["case-trace"])

    def test_mcp_tool_discovery_and_badcase_summary(self) -> None:
        service = EvalKitService(InMemoryRepository())
        server = EvalKitMcpServer(service)
        listed = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = listed["result"]["tools"]
        self.assertEqual(
            ["evaluate_rag", "explain_retrieval", "get_badcase_summary"],
            [item["name"] for item in tools],
        )
        summary = server.call_tool("get_badcase_summary", {})
        self.assertFalse(summary.get("isError", False))
        self.assertIn('"count": 0', summary["content"][0]["text"])
        unknown = server.call_tool("unknown_tool", {})
        self.assertTrue(unknown["isError"])
        server.close()

    def test_p3_01_mcp_auth_validation_timeout_and_audit(self) -> None:
        class SlowEvalKitService(EvalKitService):
            def get_badcase_summary(self, *_args, **_kwargs):
                time.sleep(0.05)
                return {}

        service = SlowEvalKitService(InMemoryRepository())
        server = EvalKitMcpServer(
            service,
            McpSecurityConfig(
                api_token="secret-token",
                timeout_seconds=0.005,
                allowed_tools=frozenset({"get_badcase_summary"}),
            ),
        )
        unauthorized = server.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        self.assertTrue(unauthorized["result"]["isError"])
        listed = server.dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            auth_token="secret-token",
        )
        self.assertEqual(
            ["get_badcase_summary"],
            [item["name"] for item in listed["result"]["tools"]],
        )
        invalid = server.call_tool(
            "get_badcase_summary",
            {"unexpected": "value"},
            auth_token="secret-token",
        )
        self.assertTrue(invalid["isError"])
        timed_out = server.call_tool(
            "get_badcase_summary", {}, auth_token="secret-token"
        )
        error_payload = json.loads(timed_out["content"][0]["text"])
        self.assertEqual("tool_timeout", error_payload["code"])
        self.assertTrue(error_payload["retryable"])
        self.assertGreaterEqual(len(service.list_audit_events()), 3)
        self.assertTrue(server.dispatch(["not", "an", "object"])["result"]["isError"])
        invalid_params = server.dispatch(
            {"id": 3, "method": "tools/call", "params": []},
            auth_token="secret-token",
        )
        self.assertTrue(invalid_params["result"]["isError"])
        invalid_token = server.dispatch({"method": "tools/list"}, auth_token=123)
        self.assertTrue(invalid_token["result"]["isError"])
        server.close()

    def test_mcp_redacts_sensitive_exception_text(self) -> None:
        class FailingService(EvalKitService):
            def get_badcase_summary(self, *_args, **_kwargs):
                raise ValueError("authorization=runtime-secret")

        server = EvalKitMcpServer(FailingService(InMemoryRepository()))
        response = server.call_tool("get_badcase_summary", {})
        payload = json.loads(response["content"][0]["text"])
        self.assertNotIn("runtime-secret", payload["message"])
        self.assertIn("[REDACTED]", payload["message"])
        server.close()

    def test_p3_02_evaluate_rag_creates_executes_and_queries_run(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("mcp-evaluate", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-mcp",
                "question": "question",
                "expected_evidence": ["mock-document#chunk-001"],
            }],
        )
        config = service.create_config("baseline")
        server = EvalKitMcpServer(service)

        created_response = server.call_tool(
            "evaluate_rag",
            {
                "operation": "create",
                "dataset_version_id": version.id,
                "config_id": config.id,
            },
        )
        created = json.loads(created_response["content"][0]["text"])
        self.assertEqual("queued", created["status"])
        run_id = created["run_id"]

        executed_response = server.call_tool(
            "evaluate_rag",
            {"operation": "execute", "eval_run_id": run_id},
        )
        executed = json.loads(executed_response["content"][0]["text"])
        self.assertEqual("succeeded", executed["status"])
        self.assertEqual(1.0, executed["summary"]["metrics"]["recall_at_k"])

        queried_response = server.call_tool(
            "evaluate_rag",
            {"operation": "get", "eval_run_id": run_id},
        )
        queried = json.loads(queried_response["content"][0]["text"])
        self.assertEqual(run_id, queried["run_id"])
        self.assertEqual("succeeded", queried["status"])

        invalid = server.call_tool(
            "evaluate_rag",
            {"operation": "create", "dataset_version_id": version.id},
        )
        self.assertTrue(invalid["isError"])

    def test_p3_03_explain_retrieval_returns_scores_filters_and_diagnosis(self) -> None:
        class ExplainableAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(
                    request_id=request_id,
                    answer="answer",
                    citations=["doc-a#chunk-1"],
                    retrievals=[
                        RetrievedChunk(
                            "chunk-1",
                            "doc-a",
                            0.91,
                            1,
                            {"rerank_score": 0.97},
                        ),
                        RetrievedChunk(
                            "chunk-2",
                            "doc-b",
                            0.82,
                            2,
                            {
                                "rerank_score": 0.31,
                                "filtered": True,
                                "filter_reason": "department mismatch",
                            },
                        ),
                    ],
                    events=[
                        {"type": "retrieval", "name": "hybrid-search"},
                        {"type": "rerank", "name": "bge-reranker"},
                    ],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("explain", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-explain",
                "question": "question",
                "expected_evidence": ["doc-a#chunk-1", "doc-c#chunk-3"],
            }],
        )
        config = service.create_config("baseline", retrieval_k=2)
        run = service.create_eval_run(version.id, config.id, "explainable")
        service.execute_eval_run(run.id, ExplainableAdapter())
        server = EvalKitMcpServer(service)
        response = server.call_tool(
            "explain_retrieval",
            {"eval_run_id": run.id, "case_id": "case-explain"},
        )
        explained = json.loads(response["content"][0]["text"])
        self.assertEqual(0.97, explained["candidates"][0]["rerank_score"])
        self.assertTrue(explained["candidates"][1]["filtered"])
        self.assertEqual(
            "department mismatch", explained["candidates"][1]["filter_reason"]
        )
        self.assertEqual(
            ["doc-c#chunk-3"], explained["missing_expected_evidence"]
        )
        self.assertEqual(2, len(explained["trace"]))

    def test_p3_04_badcase_summary_filters_version_tags_and_severity(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("badcase-summary", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [
                {
                    "id": "case-finance",
                    "question": "finance question",
                    "tags": ["finance", "policy"],
                },
                {
                    "id": "case-hr",
                    "question": "hr question",
                    "tags": ["hr", "policy"],
                },
            ],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "mock")
        service.create_badcase(
            run.id,
            "case-finance",
            "wrong_retrieval",
            "finance issue",
            severity="high",
        )
        service.create_badcase(
            run.id,
            "case-hr",
            "no_retrieval",
            "hr issue",
            severity="low",
        )
        response = EvalKitMcpServer(service).call_tool(
            "get_badcase_summary",
            {
                "dataset_version_id": version.id,
                "tags": ["finance", "policy"],
                "tag_match": "all",
                "severity": "high",
            },
        )
        summary = json.loads(response["content"][0]["text"])
        self.assertEqual(1, summary["count"])
        self.assertEqual({"high": 1}, summary["by_severity"])
        self.assertEqual({version.id: 1}, summary["by_dataset_version"])
        self.assertEqual(1, summary["by_tag"]["finance"])
        self.assertEqual("case-finance", summary["items"][0]["case_id"])

    def test_p3_05_dify_adapter_maps_blocking_response_end_to_end(self) -> None:
        class FakeResponse:
            def __init__(self, payload: bytes):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.payload

        root = Path(__file__).resolve().parents[1]
        payload = (root / "examples" / "dify_blocking_response.json").read_bytes()
        adapter = DifyChatAdapter(
            "https://dify.example.test/v1",
            "runtime-secret",
            user="evalkit-test",
        )
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("dify", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-dify",
                "question": "policy question",
                "expected_evidence": ["policy-document#chunk-001"],
            }],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "dify")
        with patch("app.dify_adapter.urlopen", return_value=FakeResponse(payload)) as mocked:
            summary = service.execute_eval_run(run.id, adapter)

        request = mocked.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("blocking", request_payload["response_mode"])
        self.assertEqual("policy question", request_payload["query"])
        self.assertEqual("Bearer runtime-secret", request.get_header("Authorization"))
        self.assertEqual(1.0, summary["metrics"]["recall_at_k"])
        result = service.list_case_results(run.id)[0]
        self.assertEqual(["policy-document#chunk-001"], result.citations)
        self.assertEqual("Synthetic Policy", result.retrieval_candidates[0]["metadata"]["document_name"])

    def test_p3_06_ragflow_adapter_maps_reference_chunks_end_to_end(self) -> None:
        class FakeResponse:
            def __init__(self, payload: bytes):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.payload

        root = Path(__file__).resolve().parents[1]
        payload = (root / "examples" / "ragflow_chat_completion.json").read_bytes()
        adapter = RagFlowChatAdapter(
            "https://ragflow.example.test",
            "runtime-secret",
            "chat-id",
        )
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("ragflow", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-ragflow",
                "question": "manual question",
                "expected_evidence": ["ragflow-document#chunk-001"],
            }],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "ragflow")
        with patch("app.ragflow_adapter.urlopen", return_value=FakeResponse(payload)) as mocked:
            summary = service.execute_eval_run(run.id, adapter)

        request = mocked.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertFalse(request_payload["stream"])
        self.assertTrue(request_payload["extra_body"]["reference"])
        self.assertIn("/api/v1/openai/chat-id/chat/completions", request.full_url)
        self.assertEqual(1.0, summary["metrics"]["recall_at_k"])
        result = service.list_case_results(run.id)[0]
        metadata = result.retrieval_candidates[0]["metadata"]
        self.assertEqual(0.91, metadata["vector_similarity"])
        self.assertEqual("Synthetic Manual", metadata["document_name"])

    def test_p4_app_metadata_and_readiness(self) -> None:
        app = create_app()
        routes = {route.path for route in app.routes}
        self.assertEqual("0.5.0", app.version)
        self.assertIn("/health", routes)
        self.assertIn("/ready", routes)

    def test_p5_benchmark_uses_synthetic_data(self) -> None:
        result = run_benchmark(case_count=2, rounds=2)
        self.assertEqual(2, result["case_count"])
        self.assertEqual(2, result["rounds"])
        self.assertGreaterEqual(result["mean_run_ms"], 0)

    def test_p6_http_adapter_retries_and_parses_contract(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "answer": "policy answer",
                    "citations": ["doc-a#chunk-1"],
                    "retrievals": [{"document_id": "doc-a", "chunk_id": "chunk-1", "score": 0.9}],
                    "events": [{"type": "retrieval", "latency_ms": 3}],
                }).encode("utf-8")

        adapter = HttpTargetAgentAdapter("https://example.test/agent", retries=1)
        with patch("app.http_adapter.urlopen", side_effect=[URLError("temporary"), FakeResponse()]) as mocked:
            result = adapter.invoke("question", request_id="run-1:case-1")
        self.assertEqual(2, mocked.call_count)
        self.assertEqual("policy answer", result.answer)
        self.assertEqual("doc-a", result.retrievals[0].document_id)
        with self.assertRaises(AdapterContractError):
            HttpTargetAgentAdapter._parse_response({"answer": 1}, "request")

    def test_p2_01_trace_event_standard_maps_aliases_and_preserves_trace_id(self) -> None:
        events = normalize_trace_events(
            [
                {"event": "retrieval_completed", "trace_id": "trace-abc", "source": "langgraph"},
                {"type": "reranker"},
                {"type": "tool_call"},
                {"type": "unknown_stage", "error": "failed"},
            ],
            eval_run_id="run-1",
            case_id="case-1",
        )
        self.assertEqual(
            ["retrieval", "rerank", "tool", "error"],
            [item.event_type.value for item in events],
        )
        self.assertEqual("trace-abc", events[0].trace_id)
        self.assertEqual("langgraph", events[0].source)
        self.assertEqual("unknown_stage", events[3].source_event_type)

    def test_p2_02_langgraph_callback_adapter_collects_node_events(self) -> None:
        def fake_graph_invoke(payload, *, config):
            self.assertEqual("question", payload["question"])
            self.assertEqual("run-x:case-y", config["configurable"]["thread_id"])
            callback = config["callbacks"][0]
            callback.on_retriever_start({"name": "retrieve_docs"}, "question", run_id="retrieve-1")
            callback.on_retriever_end(["doc"], run_id="retrieve-1")
            callback.on_llm_start({"name": "answer_llm"}, ["prompt"], run_id="llm-1")
            callback.on_llm_end("answer", run_id="llm-1")
            return {
                "answer": "answer",
                "citations": ["doc-a#chunk-1"],
                "retrievals": [{"document_id": "doc-a", "chunk_id": "chunk-1", "score": 0.9}],
            }

        result = LangGraphCallbackAdapter(fake_graph_invoke).invoke("question", request_id="run-x:case-y")
        self.assertEqual("answer", result.answer)
        self.assertEqual("doc-a", result.retrievals[0].document_id)
        self.assertEqual(["retrieval", "retrieval", "llm", "llm"], [item["type"] for item in result.events])

    def test_p2_03_langfuse_export_links_trace_usage_and_cost(self) -> None:
        class FakeObservation:
            def __init__(self, record):
                self.record = record

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self, **kwargs):
                self.record["update"] = kwargs

        class FakeLangfuseClient:
            def __init__(self):
                self.records = []
                self.flushed = False

            def start_as_current_observation(self, **kwargs):
                record = {"start": kwargs}
                self.records.append(record)
                return FakeObservation(record)

            def flush(self):
                self.flushed = True

        class LlmAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(
                    request_id=request_id,
                    answer="answer",
                    citations=["doc#chunk"],
                    retrievals=[RetrievedChunk("chunk", "doc", 1.0, 1)],
                    events=[{
                        "type": "llm",
                        "name": "answer-model",
                        "trace_id": "eval-trace",
                        "input": "prompt",
                        "output": "answer",
                    }],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("langfuse-test", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{"id": "case-lf", "question": "question", "expected_evidence": ["doc#chunk"]}],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "llm")
        service.execute_eval_run(run.id, LlmAdapter())
        client = FakeLangfuseClient()
        trace_id = service.export_case_trace(
            run.id,
            "case-lf",
            LangfuseTraceExporter(client),
            model="qwen-demo",
            usage_details={"input": 10, "output": 5},
            cost_details={"total": 0.001},
        )
        self.assertEqual(32, len(trace_id))
        self.assertTrue(client.flushed)
        generation = next(item for item in client.records if item["start"]["as_type"] == "generation")
        self.assertEqual("qwen-demo", generation["start"]["model"])
        self.assertEqual({"input": 10, "output": 5}, generation["update"]["usage_details"])
        self.assertEqual({"total": 0.001}, generation["update"]["cost_details"])

    def test_p2_04_badcase_workbench_tracks_triage_and_history(self) -> None:
        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("badcase-workbench", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{"id": "case-bad", "question": "question"}],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "mock")
        badcase = service.create_badcase(
            run.id,
            "case-bad",
            "wrong_retrieval",
            "irrelevant evidence",
            severity="high",
        )
        triaged = service.update_badcase(
            badcase.id,
            BadcaseStatus.TRIAGED,
            root_cause="metadata filter mismatch",
            owner="rag-owner",
            responsible_version="retrieval-v2",
            manual_conclusion="reproducible",
            actor="reviewer",
        )
        self.assertEqual("retrieval-v2", triaged.responsible_version)
        self.assertEqual("reproducible", triaged.manual_conclusion)
        filtered = service.list_badcases(status="triaged", severity="high", owner="rag-owner")
        self.assertEqual([badcase.id], [item.id for item in filtered])
        with self.assertRaises(ValueError):
            service.update_badcase(badcase.id, BadcaseStatus.FIXED, actor="reviewer")
        service.update_badcase(
            badcase.id,
            BadcaseStatus.FIXED,
            resolution="fixed metadata mapping",
            actor="reviewer",
        )
        detail = service.get_badcase_detail(badcase.id)
        self.assertEqual(3, len(detail["activities"]))
        self.assertEqual("fixed", detail["badcase"]["status"].value)

    def test_p2_05_regression_suite_persists_regressed_comparison(self) -> None:
        class RetrievalAdapter:
            def __init__(self, document_id: str):
                self.document_id = document_id

            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                chunk = RetrievedChunk("chunk-1", self.document_id, 0.9, 1)
                return AdapterResponse(
                    request_id=request_id,
                    answer="answer",
                    citations=[f"{self.document_id}#chunk-1"],
                    retrievals=[chunk],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("regression", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{"id": "case-reg", "question": "question", "expected_evidence": ["good#chunk-1"]}],
        )
        config = service.create_config("baseline")
        baseline_run = service.create_eval_run(version.id, config.id, "good")
        candidate_run = service.create_eval_run(version.id, config.id, "bad")
        service.execute_eval_run(baseline_run.id, RetrievalAdapter("good"))
        service.execute_eval_run(candidate_run.id, RetrievalAdapter("wrong"))
        badcase = service.create_badcase(
            baseline_run.id, "case-reg", "wrong_retrieval", "regression seed"
        )
        suite = service.create_regression_suite_from_badcases("suite", [badcase.id])
        comparison = service.compare_regression_runs(
            suite.id, baseline_run.id, candidate_run.id
        )
        self.assertEqual("regressed", comparison["comparison"]["case-reg"])
        self.assertEqual(1, comparison["summary"]["regressed"])
        stored = service.get_regression_comparison(comparison["comparison_id"])
        self.assertEqual(version.id, stored["candidate_dataset_version_id"])
        self.assertEqual(1, len(service.list_regression_comparisons(suite.id)))

    def test_p2_06_answer_quality_and_human_review(self) -> None:
        class StructuredAdapter:
            def invoke(self, question: str, request_id: str) -> AdapterResponse:
                return AdapterResponse(
                    request_id=request_id,
                    answer='{"policy_code":"POL-123","amount":100}',
                    citations=["policy#chunk-1"],
                    retrievals=[RetrievedChunk("chunk-1", "policy", 0.95, 1)],
                )

        service = EvalKitService(InMemoryRepository())
        dataset = service.create_dataset("quality", "tester")
        version = service.import_dataset_version(
            dataset.id,
            [{
                "id": "case-quality",
                "question": "question",
                "expected_evidence": ["policy#chunk-1"],
                "metadata": {
                    "required_keywords": ["POL-123"],
                    "required_patterns": [r"POL-\d+"],
                    "require_citations": True,
                    "answer_json_schema": {
                        "type": "object",
                        "required": ["policy_code", "amount"],
                        "properties": {
                            "policy_code": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                },
            }],
        )
        config = service.create_config("baseline")
        run = service.create_eval_run(version.id, config.id, "structured")
        service.execute_eval_run(run.id, StructuredAdapter())
        result = service.list_case_results(run.id)[0]
        self.assertTrue(result.quality_checks["keyword_pass"])
        self.assertTrue(result.quality_checks["regex_pass"])
        self.assertTrue(result.quality_checks["json_schema_pass"])
        self.assertTrue(result.quality_checks["citation_validity_pass"])

        reviewed = service.review_case_result(
            run.id, "case-quality", 4.5, "medical-reviewer", "acceptable"
        )
        self.assertEqual(4.5, reviewed.human_score)
        self.assertEqual("medical-reviewer", reviewed.reviewed_by)
        invalid = validate_answer(
            "answer",
            ["unknown#chunk"],
            {"require_citations": True},
            ["known#chunk"],
        )
        self.assertFalse(invalid["citation_validity_pass"])

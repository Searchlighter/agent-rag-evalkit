from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from app.http_adapter import HttpTargetAgentAdapter
from app.ingestion import load_jsonl_cases
from app.mock_rag_service import SYNTHETIC_CHUNKS, build_mock_response
from scripts.demo_http_eval import load_dataset, run_demo


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class P402DemoTests(unittest.TestCase):
    def test_synthetic_dataset_is_utf8_and_matches_mock_evidence(self) -> None:
        cases = load_jsonl_cases(ROOT / "sample_data" / "enterprise_eval_cases.jsonl")
        knowledge = json.loads(
            (ROOT / "sample_data" / "mock_knowledge_base.json").read_text(encoding="utf-8")
        )
        case_evidence = {item for case in cases for item in case["expected_evidence"]}
        file_evidence = {f'{item["document_id"]}#{item["chunk_id"]}' for item in knowledge}
        service_evidence = {item.evidence_id for item in SYNTHETIC_CHUNKS}

        self.assertEqual(4, len(cases))
        self.assertEqual(case_evidence, file_evidence)
        self.assertEqual(case_evidence, service_evidence)
        self.assertIn("差旅报销", cases[0]["question"])

    def test_mock_rag_response_matches_http_adapter_contract(self) -> None:
        raw = build_mock_response("采购合同需要经过哪些审批步骤？", "request-1")
        parsed = HttpTargetAgentAdapter._parse_response(raw, "request-1")

        self.assertEqual("request-1", parsed.request_id)
        self.assertEqual("procurement-policy#contract", parsed.citations[0])
        self.assertEqual("procurement-policy", parsed.retrievals[0].document_id)
        self.assertEqual("synthetic", parsed.retrievals[0].metadata["source"])

    def test_http_demo_runs_dataset_end_to_end_with_mock_transport(self) -> None:
        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            return FakeResponse(build_mock_response(body["question"], body["request_id"]))

        with patch("app.http_adapter.urlopen", side_effect=fake_urlopen):
            output = run_demo()

        self.assertEqual("succeeded", output["summary"]["status"])
        self.assertEqual(4, output["summary"]["completed_case_count"])
        self.assertEqual(1.0, output["summary"]["metrics"]["recall_at_k"])
        self.assertTrue(all(item["status"] == "succeeded" for item in output["results"]))
        self.assertTrue(
            all(item["quality_checks"]["keyword_pass"] for item in output["results"])
        )

    def test_http_demo_selects_csv_endpoint_and_environment_token(self) -> None:
        captured_requests = []

        def fake_urlopen(request, timeout):
            captured_requests.append((request, timeout))
            body = json.loads(request.data.decode("utf-8"))
            return FakeResponse(build_mock_response(body["question"], body["request_id"]))

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "selected.csv"
            dataset.write_text(
                "id,question,expected_answers,expected_evidence,tags\n"
                "csv-1,员工如何申请差旅报销？,提交报销单,expense-policy#travel,expense\n",
                encoding="utf-8",
            )
            with (
                patch.dict("app.http_adapter.environ", {"DEMO_TOKEN": "safe-secret"}),
                patch("app.http_adapter.urlopen", side_effect=fake_urlopen),
            ):
                output = run_demo(
                    "https://agent.example.test/query",
                    dataset,
                    bearer_token_env="DEMO_TOKEN",
                    timeout_seconds=3,
                    retries=0,
                    retrieval_k=1,
                )

        request, timeout = captured_requests[0]
        self.assertEqual("https://agent.example.test/query", request.full_url)
        self.assertEqual("Bearer safe-secret", request.get_header("Authorization"))
        self.assertEqual(3, timeout)
        self.assertEqual(1, output["configuration"]["case_count"])
        self.assertTrue(output["configuration"]["authenticated"])
        self.assertNotIn("safe-secret", json.dumps(output, ensure_ascii=False))

    def test_http_demo_rejects_missing_empty_or_unsupported_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            unsupported = root / "cases.txt"
            unsupported.write_text("question", encoding="utf-8")

            for path in (root / "missing.jsonl", empty, unsupported):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    load_dataset(path)


if __name__ == "__main__":
    unittest.main()

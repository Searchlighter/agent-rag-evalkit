"""Reference RAG 索引、排序和 HTTP 契约测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.http_adapter import HttpTargetAgentAdapter
from app.reference_rag import ReferenceRagIndex, build_reference_response, create_reference_rag_app


ROOT = Path(__file__).resolve().parents[1]


class ReferenceRagTests(unittest.TestCase):
    """验证真实索引能从外部语料执行两种检索策略。"""

    @classmethod
    def setUpClass(cls) -> None:
        """为全部测试复用同一个外部语料索引。"""
        cls.index = ReferenceRagIndex.from_json()

    def test_corpus_and_evaluation_dataset_are_external_utf8_files(self) -> None:
        """语料和评测集应独立于 Python 源码并包含明确数据范围。"""
        corpus = json.loads((ROOT / "app/data/reference_corpus.json").read_text(encoding="utf-8"))
        cases = [
            json.loads(line)
            for line in (ROOT / "sample_data/reference_eval_cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(12, len(corpus))
        self.assertEqual(24, len(cases))
        self.assertIn("差旅费用管理办法", corpus[0]["title"])

    def test_hybrid_recovers_synonym_query_missed_by_bm25(self) -> None:
        """混合检索应修复协议会签这一同义表达的 BM25 漏召回。"""
        expected = "procurement-policy#contract"
        baseline = [item["evidence_id"] for item in self.index.search("协议会签涉及哪些角色？", "bm25", 3)]
        candidate = [item["evidence_id"] for item in self.index.search("协议会签涉及哪些角色？", "hybrid", 3)]
        self.assertNotIn(expected, baseline)
        self.assertEqual(expected, candidate[0])

    def test_response_matches_http_adapter_contract(self) -> None:
        """Reference RAG 输出应可直接被通用 HTTP Adapter 解析。"""
        raw = build_reference_response(
            self.index,
            "收到钓鱼邮件应该怎么上报？",
            "request-001",
            "hybrid",
        )
        parsed = HttpTargetAgentAdapter._parse_response(raw, "request-001")
        self.assertEqual("security-handbook#phishing", parsed.citations[0])
        self.assertEqual("hybrid", parsed.retrievals[0].metadata["strategy"])
        self.assertGreater(parsed.retrievals[0].metadata["bm25_score"], 0)

    def test_fastapi_registers_health_and_strategy_endpoints(self) -> None:
        """服务应公开健康检查以及两个独立策略端点。"""
        app = create_reference_rag_app()
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/v1/query/bm25", paths)
        self.assertIn("/v1/query/hybrid", paths)

    def test_committed_comparison_report_is_stable_and_auditable(self) -> None:
        """提交的报告应固定、可复查，且不泄露本机绝对路径。"""
        report_path = ROOT / "reports" / "reference-rag-comparison.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        serialized = json.dumps(report, ensure_ascii=False)

        self.assertEqual(24, report["dataset"]["case_count"])
        self.assertEqual(12, report["corpus"]["chunk_count"])
        self.assertEqual(0.9167, report["baseline"]["summary"]["metrics"]["recall_at_k"])
        self.assertEqual(1.0, report["candidate"]["summary"]["metrics"]["recall_at_k"])
        self.assertNotIn("run_id", serialized)
        self.assertNotIn("dataset_version_id", serialized)
        self.assertNotIn(str(ROOT), serialized)


if __name__ == "__main__":
    unittest.main()

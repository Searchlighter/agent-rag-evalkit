"""通过 HTTP Adapter 对比本地 Reference RAG 的 BM25 与混合检索。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.http_adapter import HttpTargetAgentAdapter
from app.ingestion import load_jsonl_cases
from app.repository import InMemoryRepository
from app.service import EvalKitService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "sample_data" / "reference_eval_cases.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "reference-rag-comparison.json"


def _execute(endpoint: str, cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """通过指定 HTTP 端点执行完整评测并提取失败样本。"""
    service = EvalKitService(InMemoryRepository())
    dataset = service.create_dataset("reference-rag-evaluation", "comparison-script")
    version = service.import_dataset_version(dataset.id, cases, "comparison-script")
    config = service.create_config(f"{name}-top3", retrieval_k=3)
    run = service.create_eval_run(version.id, config.id, name)
    summary = service.execute_eval_run(
        run.id,
        HttpTargetAgentAdapter(endpoint=endpoint, timeout_seconds=5, retries=1),
    )
    summary = {
        key: value
        for key, value in summary.items()
        if key not in {"run_id", "dataset_version_id"}
    }
    failures = []
    for result in service.list_case_results(run.id):
        if result.metrics.get("recall_at_k") == 0:
            case = next(item for item in cases if item["id"] == result.case_id)
            failures.append(
                {
                    "case_id": result.case_id,
                    "question": case["question"],
                    "expected_evidence": case["expected_evidence"],
                    "retrieval_ids": result.retrieval_ids,
                    "diagnosis": service.diagnose_case(run.id, result.case_id),
                }
            )
    return {"summary": summary, "failures": failures}


def compare(
    base_url: str = "http://127.0.0.1:8002",
    dataset_path: str | Path = DEFAULT_DATASET,
) -> dict[str, Any]:
    """在相同数据集和 Top-K 下执行两种策略并计算指标差值。"""
    cases = load_jsonl_cases(dataset_path)
    baseline = _execute(f"{base_url}/v1/query/bm25", cases, "reference-bm25")
    candidate = _execute(f"{base_url}/v1/query/hybrid", cases, "reference-hybrid")
    baseline_metrics = baseline["summary"]["metrics"]
    candidate_metrics = candidate["summary"]["metrics"]
    return {
        "schema": "agent_rag_evalkit/reference_comparison/v1",
        "dataset": {
            "path": (
                str(Path(dataset_path).resolve().relative_to(ROOT).as_posix())
                if Path(dataset_path).resolve().is_relative_to(ROOT)
                else str(Path(dataset_path).as_posix())
            ),
            "case_count": len(cases),
            "data_classification": "synthetic-public",
        },
        "corpus": {
            "path": "app/data/reference_corpus.json",
            "chunk_count": 12,
            "data_classification": "synthetic-public",
        },
        "configuration": {"top_k": 3, "baseline": "BM25", "candidate": "BM25 + query expansion + hashed n-gram vector"},
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: round((candidate_metrics.get(key) or 0) - (baseline_metrics.get(key) or 0), 4)
            for key in ("recall_at_k", "mrr", "citation_coverage")
        },
        "limitations": [
            "语料与问题均为公开可提交的人工合成数据，不代表生产业务分布。",
            "哈希 n-gram 向量用于离线可复现演示，不等同于神经网络 Embedding。",
            "回答采用首条证据抽取，不评估生成式模型的语义质量。",
        ],
    }


def main() -> None:
    """解析服务地址、数据集和报告路径并保存对比结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    result = compare(arguments.base_url, arguments.dataset)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved report: {arguments.output}")


if __name__ == "__main__":
    main()

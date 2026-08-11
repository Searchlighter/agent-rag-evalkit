"""通过 HTTP Adapter 对独立合成 RAG 服务执行端到端评测。"""

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
DEFAULT_DATASET = ROOT / "sample_data" / "enterprise_eval_cases.jsonl"


def run_demo(
    endpoint: str = "http://127.0.0.1:8001/v1/query",
    dataset_path: str | Path = DEFAULT_DATASET,
) -> dict[str, Any]:
    """导入本地数据集、执行评测并返回可序列化的摘要与逐条结果。"""
    cases = load_jsonl_cases(dataset_path)
    service = EvalKitService(InMemoryRepository())
    dataset = service.create_dataset("synthetic-enterprise-knowledge", "p4-demo")
    version = service.import_dataset_version(dataset.id, cases, actor_id="p4-demo")
    config = service.create_config("http-mock-baseline", retrieval_k=2)
    run = service.create_eval_run(version.id, config.id, "synthetic-http-rag")
    summary = service.execute_eval_run(
        run.id,
        HttpTargetAgentAdapter(endpoint=endpoint, timeout_seconds=5, retries=1),
    )
    return {
        "summary": summary,
        "results": [
            {
                "case_id": item.case_id,
                "status": item.status,
                "answer": item.answer,
                "retrieval_ids": item.retrieval_ids,
                "metrics": item.metrics,
                "quality_checks": item.quality_checks,
            }
            for item in service.list_case_results(run.id)
        ],
    }


def main() -> None:
    """解析命令行参数，并在服务不可用时给出可操作提示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/query")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    arguments = parser.parse_args()
    try:
        output = run_demo(arguments.endpoint, arguments.dataset)
    except Exception as error:
        raise SystemExit(
            f"Demo failed: {error}\n"
            "Start the mock service first: "
            "uvicorn app.mock_rag_service:app --port 8001"
        ) from error
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

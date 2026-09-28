"""通过 HTTP Adapter 对独立合成 RAG 服务执行端到端评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.http_adapter import HttpTargetAgentAdapter
from app.ingestion import load_csv_cases, load_jsonl_cases
from app.repository import InMemoryRepository
from app.service import EvalKitService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "sample_data" / "enterprise_eval_cases.jsonl"
DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1/query"


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """按扩展名加载用户选择的 JSONL 或 CSV 数据集。"""
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise ValueError(f"dataset file does not exist: {dataset_path}")
    loaders = {".jsonl": load_jsonl_cases, ".csv": load_csv_cases}
    try:
        loader = loaders[dataset_path.suffix.lower()]
    except KeyError as error:
        raise ValueError("dataset must be a .jsonl or .csv file") from error
    cases = loader(dataset_path)
    if not cases:
        raise ValueError("dataset must contain at least one case")
    return cases


def run_demo(
    endpoint: str = DEFAULT_ENDPOINT,
    dataset_path: str | Path = DEFAULT_DATASET,
    *,
    bearer_token_env: str | None = None,
    timeout_seconds: float = 5,
    retries: int = 1,
    retrieval_k: int = 2,
) -> dict[str, Any]:
    """导入本地数据集、执行评测并返回可序列化的摘要与逐条结果。"""
    cases = load_dataset(dataset_path)
    service = EvalKitService(InMemoryRepository())
    selected_path = Path(dataset_path).expanduser().resolve()
    dataset = service.create_dataset(selected_path.stem, "http-demo")
    version = service.import_dataset_version(dataset.id, cases, actor_id="p4-demo")
    config = service.create_config("http-evaluation", retrieval_k=retrieval_k)
    run = service.create_eval_run(version.id, config.id, "synthetic-http-rag")
    summary = service.execute_eval_run(
        run.id,
        HttpTargetAgentAdapter(
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            retries=retries,
            bearer_token_env=bearer_token_env,
        ),
    )
    return {
        "configuration": {
            "endpoint": endpoint,
            "dataset": str(selected_path),
            "case_count": len(cases),
            "retrieval_k": retrieval_k,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "authenticated": bearer_token_env is not None,
        },
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
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="目标查询接口 URL")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL 或 CSV 评测集"
    )
    parser.add_argument(
        "--bearer-token-env",
        help="保存 Bearer Token 的环境变量名；不会读取命令行明文 Token",
    )
    parser.add_argument("--timeout", type=float, default=5, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, choices=range(0, 6), default=1)
    parser.add_argument("--retrieval-k", type=int, default=2, help="召回指标的 K 值")
    arguments = parser.parse_args()
    try:
        output = run_demo(
            arguments.endpoint,
            arguments.dataset,
            bearer_token_env=arguments.bearer_token_env,
            timeout_seconds=arguments.timeout,
            retries=arguments.retries,
            retrieval_k=arguments.retrieval_k,
        )
    except Exception as error:
        raise SystemExit(
            f"Demo failed: {error}\n"
            "Start the mock service first: "
            "uvicorn app.mock_rag_service:app --port 8001"
        ) from error
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

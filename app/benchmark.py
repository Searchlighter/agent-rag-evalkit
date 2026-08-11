"""Reproducible synthetic benchmark; never treat it as a production capacity claim."""

from __future__ import annotations

import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from math import ceil
from statistics import mean
from time import perf_counter
from typing import Any

from .adapter_contract import MockRagAdapter
from .repository import InMemoryRepository
from .service import EvalKitService


BENCHMARK_SCHEMA = "agent_rag_evalkit_benchmark/v1"


def _execute_run(case_count: int, run_index: int) -> float:
    service = EvalKitService(InMemoryRepository())
    dataset = service.create_dataset(f"benchmark-{run_index}", "benchmark")
    cases = [
        {
            "id": f"case-{run_index}-{case_index}",
            "question": f"synthetic question {case_index}",
            "expected_evidence": ["mock-document#chunk-001"],
            "metadata": {"require_citations": True},
        }
        for case_index in range(case_count)
    ]
    version = service.import_dataset_version(dataset.id, cases)
    config = service.create_config("benchmark", retrieval_k=5)
    run = service.create_eval_run(version.id, config.id, "mock-rag")
    started = perf_counter()
    summary = service.execute_eval_run(run.id, MockRagAdapter())
    duration_ms = (perf_counter() - started) * 1000
    if summary["status"] != "succeeded" or summary["completed_case_count"] != case_count:
        raise RuntimeError("synthetic benchmark run did not complete successfully")
    return duration_ms


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def run_benchmark(
    case_count: int = 20,
    rounds: int = 5,
    concurrency: int = 1,
    warmup_rounds: int = 1,
) -> dict[str, Any]:
    """并发执行独立内存评测，记录环境、延迟、吞吐和实际成本边界。"""
    if case_count <= 0 or rounds <= 0:
        raise ValueError("case_count and rounds must be positive")
    if concurrency <= 0 or concurrency > rounds:
        raise ValueError("concurrency must be positive and cannot exceed rounds")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds cannot be negative")

    for index in range(warmup_rounds):
        _execute_run(case_count, -(index + 1))

    wall_started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        durations_ms = list(
            executor.map(lambda index: _execute_run(case_count, index), range(rounds))
        )
    wall_ms = (perf_counter() - wall_started) * 1000
    total_cases = case_count * rounds
    machine = platform.machine() or os.environ.get("PROCESSOR_ARCHITECTURE") or "unknown"
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or machine
    latency = {
        "min_run_ms": round(min(durations_ms), 3),
        "mean_run_ms": round(mean(durations_ms), 3),
        "p50_run_ms": round(_nearest_rank(durations_ms, 0.50), 3),
        "p95_run_ms": round(_nearest_rank(durations_ms, 0.95), 3),
        "p99_run_ms": round(_nearest_rank(durations_ms, 0.99), 3),
        "max_run_ms": round(max(durations_ms), 3),
        "mean_case_ms": round(mean(durations_ms) / case_count, 3),
        "wall_time_ms": round(wall_ms, 3),
    }
    throughput = round(total_cases / (wall_ms / 1000), 2) if wall_ms > 0 else 0.0
    return {
        "schema": BENCHMARK_SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": machine,
            "processor": processor,
            "logical_cpu_count": os.cpu_count(),
            "executable": sys.executable,
        },
        "workload": {
            "mode": "in_memory_synthetic",
            "adapter": "MockRagAdapter",
            "storage": "InMemoryRepository",
            "case_count_per_run": case_count,
            "rounds": rounds,
            "warmup_rounds": warmup_rounds,
            "concurrency": concurrency,
            "total_measured_cases": total_cases,
            "retrieval_k": 5,
            "external_network_calls": False,
        },
        "latency": latency,
        "throughput": {"cases_per_second": throughput},
        "cost": {
            "currency": "CNY",
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "measured_api_cost": 0.0,
            "reason": "MockRagAdapter performs no model or external API calls.",
        },
        "limitations": [
            "Measures Python in-memory orchestration with synthetic data only.",
            "Concurrent workers use independent EvalKitService instances.",
            "Does not measure HTTP, database, vector search, LLM latency, or production load.",
        ],
        "case_count": case_count,
        "rounds": rounds,
        "mean_run_ms": latency["mean_run_ms"],
        "p95_run_ms": latency["p95_run_ms"],
        "mean_case_ms": latency["mean_case_ms"],
    }

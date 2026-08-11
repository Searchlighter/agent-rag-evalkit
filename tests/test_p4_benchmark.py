import json
import unittest
from pathlib import Path

from app.benchmark import BENCHMARK_SCHEMA, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class P405BenchmarkTests(unittest.TestCase):
    def test_benchmark_records_environment_workload_latency_throughput_and_cost(self) -> None:
        result = run_benchmark(case_count=2, rounds=2, concurrency=2, warmup_rounds=0)

        self.assertEqual(BENCHMARK_SCHEMA, result["schema"])
        self.assertTrue(result["environment"]["python_version"])
        self.assertTrue(result["environment"]["machine"])
        self.assertTrue(result["environment"]["processor"])
        self.assertEqual(2, result["workload"]["concurrency"])
        self.assertEqual(4, result["workload"]["total_measured_cases"])
        self.assertIn("p50_run_ms", result["latency"])
        self.assertIn("p95_run_ms", result["latency"])
        self.assertIn("p99_run_ms", result["latency"])
        self.assertGreater(result["throughput"]["cases_per_second"], 0)
        self.assertEqual(0, result["cost"]["model_calls"])
        self.assertEqual(0.0, result["cost"]["measured_api_cost"])
        self.assertFalse(result["workload"]["external_network_calls"])

    def test_benchmark_preserves_earlier_summary_fields(self) -> None:
        result = run_benchmark(case_count=2, rounds=2, warmup_rounds=0)

        self.assertEqual(2, result["case_count"])
        self.assertEqual(2, result["rounds"])
        self.assertEqual(result["latency"]["mean_run_ms"], result["mean_run_ms"])
        self.assertEqual(result["latency"]["p95_run_ms"], result["p95_run_ms"])

    def test_benchmark_rejects_invalid_concurrency_and_warmup(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(case_count=1, rounds=1, concurrency=2)
        with self.assertRaises(ValueError):
            run_benchmark(case_count=1, rounds=1, warmup_rounds=-1)

    def test_recorded_baseline_uses_versioned_schema_and_explicit_limitations(self) -> None:
        baseline = json.loads(
            (ROOT / "benchmarks" / "baseline-p4-05.json").read_text(encoding="utf-8")
        )

        self.assertEqual(BENCHMARK_SCHEMA, baseline["schema"])
        self.assertEqual(2000, baseline["workload"]["total_measured_cases"])
        self.assertEqual(4, baseline["workload"]["concurrency"])
        self.assertEqual(0.0, baseline["cost"]["measured_api_cost"])
        self.assertGreaterEqual(len(baseline["limitations"]), 3)


if __name__ == "__main__":
    unittest.main()

"""Dashboard 演示数据、指标快照和页面路由测试。"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.dashboard import bootstrap_dashboard_demo, build_dashboard_snapshot
from app.main import create_app
from app.repository import InMemoryRepository
from app.service import EvalKitService


class DashboardTests(unittest.TestCase):
    """验证可视化页面展示的数据真实来自评测服务。"""

    def test_demo_snapshot_contains_metrics_results_and_badcases(self) -> None:
        """合成演示应稳定产生六条结果与两条错误召回 Badcase。"""
        service = EvalKitService(InMemoryRepository())
        run_id = bootstrap_dashboard_demo(service)

        snapshot = build_dashboard_snapshot(service, run_id)

        self.assertEqual(6, snapshot["summary"]["completed_case_count"])
        self.assertEqual(0.6667, snapshot["summary"]["metrics"]["recall_at_k"])
        self.assertEqual(6, len(snapshot["results"]))
        self.assertEqual(2, len(snapshot["badcases"]))
        self.assertEqual(
            24, snapshot["reference_comparison"]["dataset"]["case_count"]
        )
        self.assertEqual(
            0.0833, snapshot["reference_comparison"]["delta"]["recall_at_k"]
        )
        self.assertTrue(
            all(item["category"] == "wrong_retrieval" for item in snapshot["badcases"])
        )

    def test_application_exposes_dashboard_and_snapshot_routes(self) -> None:
        """应用应同时注册可视化页面及其只读数据接口。"""
        client = TestClient(create_app())

        html_response = client.get("/dashboard")
        snapshot_response = client.get("/api/v1/dashboard/snapshot")

        self.assertEqual(200, html_response.status_code)
        self.assertIn("AgentRAG EvalKit", html_response.text)
        self.assertEqual(200, snapshot_response.status_code)
        snapshot = snapshot_response.json()
        self.assertEqual("succeeded", snapshot["summary"]["status"])


if __name__ == "__main__":
    unittest.main()

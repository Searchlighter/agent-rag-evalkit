"""Dashboard 演示数据、指标快照和页面路由测试。"""

from __future__ import annotations

import unittest

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
        routes = self._collect_routes(create_app().routes)

        self.assertIn("/dashboard", routes)
        self.assertIn("/api/v1/dashboard/snapshot", routes)
        html_response = routes["/dashboard"].endpoint()
        self.assertIn("AgentRAG EvalKit", html_response.body.decode("utf-8"))
        snapshot = routes["/api/v1/dashboard/snapshot"].endpoint()
        self.assertEqual("succeeded", snapshot["summary"]["status"])

    @classmethod
    def _collect_routes(cls, routes: list[object]) -> dict[str, object]:
        """兼容 FastAPI 旧版扁平路由与新版嵌套路由结构。"""
        collected: dict[str, object] = {}
        for route in routes:
            path = getattr(route, "path", None)
            if path is not None:
                collected[path] = route
            nested = getattr(route, "routes", None)
            if not nested:
                original_router = getattr(route, "original_router", None)
                nested = getattr(original_router, "routes", None)
            if nested:
                collected.update(cls._collect_routes(list(nested)))
        return collected


if __name__ == "__main__":
    unittest.main()

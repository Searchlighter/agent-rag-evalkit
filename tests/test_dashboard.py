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

        self.assertEqual("本地演示项目", snapshot["workspace"]["name"])
        self.assertTrue(snapshot["workspace"]["is_demo"])
        self.assertEqual(0, snapshot["workspace"]["user_run_count"])
        self.assertTrue(snapshot["demo"]["is_demo"])
        self.assertEqual("demo", snapshot["demo"]["source"])
        self.assertEqual([], snapshot["user_runs"])
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

    def test_snapshot_separates_user_runs_from_built_in_demo(self) -> None:
        service = EvalKitService(InMemoryRepository())
        demo_run_id = bootstrap_dashboard_demo(service)
        dataset = service.create_dataset("用户评测集", "user-1")
        version = service.import_dataset_version(
            dataset.id, [{"id": "user-case", "question": "真实问题"}], "user-1"
        )
        config = service.create_config("用户配置", 3)
        user_run = service.create_eval_run(
            version.id, config.id, "production-rag", model_id="release-1"
        )

        snapshot = build_dashboard_snapshot(service, demo_run_id)

        self.assertEqual(1, snapshot["workspace"]["user_run_count"])
        self.assertEqual(demo_run_id, snapshot["demo"]["summary"]["run_id"])
        self.assertNotEqual(demo_run_id, snapshot["user_runs"][0]["run_id"])
        self.assertEqual(user_run.id, snapshot["user_runs"][0]["run_id"])
        self.assertEqual("user", snapshot["user_runs"][0]["source"])
        self.assertFalse(snapshot["user_runs"][0]["is_demo"])
        self.assertEqual("用户评测集", snapshot["user_runs"][0]["dataset_name"])
        self.assertEqual("production-rag", snapshot["user_runs"][0]["adapter_id"])

    def test_application_exposes_dashboard_and_snapshot_routes(self) -> None:
        """应用应同时注册可视化页面及其只读数据接口。"""
        routes = self._collect_routes(create_app().routes)

        self.assertIn("/dashboard", routes)
        self.assertIn("/api/v1/dashboard/snapshot", routes)
        html_response = routes["/dashboard"].endpoint()
        html = html_response.body.decode("utf-8")
        self.assertIn("AgentRAG EvalKit", html)
        for navigation in ("概览", "Adapters", "数据集", "评测运行", "Badcase"):
            self.assertIn(f">{navigation}<", html)
        self.assertIn('data-page="overview"', html)
        self.assertIn('id="adapter-form"', html)
        self.assertIn("创建 Adapter", html)
        self.assertIn("data-adapter-action", html)
        self.assertIn('id="http-configuration-form"', html)
        self.assertIn("Bearer Token（环境变量）", html)
        self.assertIn("测试连接", html)
        self.assertIn('id="dataset-upload-form"', html)
        self.assertIn("上传并校验", html)
        self.assertIn("JSONL 每行必须是一个对象", html)
        self.assertIn('id="dataset-validation-errors"', html)
        self.assertIn("重复 ID", html)
        self.assertIn("编码错误", html)
        self.assertIn('id="dataset-list"', html)
        self.assertIn('id="dataset-preview"', html)
        self.assertIn("样本预览", html)
        self.assertIn("loadDatasets()", html)
        self.assertIn("还没有用户导入的数据集", html)
        self.assertIn('id="eval-run-form"', html)
        self.assertIn('id="run-dataset-version"', html)
        self.assertIn('id="run-adapter"', html)
        self.assertIn("创建评测任务", html)
        self.assertIn('id="refresh-runs"', html)
        self.assertIn("执行评测", html)
        self.assertIn("loadRuns()", html)
        self.assertIn('id="run-results-panel"', html)
        self.assertIn("查看结果", html)
        self.assertIn("逐 Case 结果", html)
        self.assertIn("召回内容", html)
        self.assertIn("暂无用户创建的评测运行", html)
        self.assertIn("用户运行", html)
        self.assertIn("内置演示运行", html)
        self.assertIn("用户运行独立展示", html)
        self.assertIn("工作台加载失败", html)
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

"""可视化 Dashboard 路由及合成演示数据装载。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .adapter_contract import MockRagAdapter
from .domain import BadcaseSeverity
from .service import EvalKitService


_DASHBOARD_HTML = Path(__file__).with_name("static") / "dashboard.html"

_DEMO_CASES: list[dict[str, Any]] = [
    {
        "id": "demo_expense",
        "question": "员工出差结束后如何申请差旅报销？",
        "expected_answers": ["提交报销单，并附发票和行程凭证。"],
        "expected_evidence": ["mock-document#chunk-001"],
        "tags": ["财务", "差旅"],
        "metadata": {"require_citations": True},
    },
    {
        "id": "demo_contract",
        "question": "采购合同需要经过哪些审批步骤？",
        "expected_answers": ["部门负责人、法务和财务依次审批。"],
        "expected_evidence": ["mock-document#chunk-001"],
        "tags": ["采购", "合同"],
        "metadata": {"require_citations": True},
    },
    {
        "id": "demo_password",
        "question": "忘记公司账号密码后应该怎么处理？",
        "expected_answers": ["通过身份平台自助重置或联系 IT 服务台。"],
        "expected_evidence": ["mock-document#chunk-001"],
        "tags": ["IT", "账号"],
        "metadata": {"require_citations": True},
    },
    {
        "id": "demo_leave",
        "question": "正式员工每年有多少天基础年假？",
        "expected_answers": ["基础年假为五个工作日。"],
        "expected_evidence": ["mock-document#chunk-001"],
        "tags": ["人事", "休假"],
        "metadata": {"require_citations": True},
    },
    {
        "id": "demo_security",
        "question": "发现疑似钓鱼邮件后如何上报？",
        "expected_answers": ["保留邮件并向安全团队报告。"],
        "expected_evidence": ["security-handbook#phishing"],
        "tags": ["安全", "邮件"],
        "metadata": {"require_citations": True},
    },
    {
        "id": "demo_remote",
        "question": "远程办公设备损坏后由谁负责维修？",
        "expected_answers": ["通过资产服务台提交维修申请。"],
        "expected_evidence": ["it-handbook#remote-device"],
        "tags": ["IT", "远程办公"],
        "metadata": {"require_citations": True},
    },
]


def bootstrap_dashboard_demo(service: EvalKitService) -> str:
    """创建一次含成功与失败样本的确定性演示运行，并返回运行ID。"""
    dataset = service.create_dataset("企业知识库演示评测集", "dashboard-demo")
    version = service.import_dataset_version(dataset.id, _DEMO_CASES, "dashboard-demo")
    config = service.create_config("Mock Baseline · Top 3", retrieval_k=3)
    run = service.create_eval_run(version.id, config.id, "mock-rag", 1.0)
    service.execute_eval_run(run.id, MockRagAdapter())

    # 将确定性检索失败转入 Badcase 工作台，供 Dashboard 展示诊断闭环。
    for result in service.list_case_results(run.id):
        if result.metrics.get("recall_at_k") == 0:
            diagnosis = service.diagnose_case(run.id, result.case_id)
            service.create_badcase(
                run.id,
                result.case_id,
                diagnosis["suggested_category"],
                diagnosis["suggestion"],
                BadcaseSeverity.HIGH,
            )
    return run.id


def build_dashboard_snapshot(service: EvalKitService, run_id: str) -> dict[str, Any]:
    """聚合页面需要的运行摘要、样本结果和 Badcase 信息。"""
    summary = service.get_run_summary(run_id)
    run = service.repository.eval_runs[run_id]
    version = service.repository.dataset_versions[run.dataset_version_id]
    cases = {item.id: item for item in version.cases}

    results: list[dict[str, Any]] = []
    for result in service.list_case_results(run_id):
        case = cases[result.case_id]
        diagnosis = service.diagnose_case(run_id, result.case_id)
        results.append(
            {
                "case_id": result.case_id,
                "question": case.question,
                "tags": case.tags,
                "status": result.status,
                "answer": result.answer,
                "citations": result.citations,
                "retrieval_ids": result.retrieval_ids,
                "metrics": result.metrics,
                "diagnosis": diagnosis,
            }
        )

    return {
        "demo_notice": "当前页面使用脱敏合成数据与确定性 Mock Adapter，仅用于展示评测工作流。",
        "summary": summary,
        "results": results,
        "badcases": [asdict(item) for item in service.list_badcases(eval_run_id=run_id)],
    }


def create_dashboard_router(service: EvalKitService, demo_run_id: str) -> APIRouter:
    """创建 Dashboard 页面及其只读快照接口。"""
    router = APIRouter()

    @router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        """返回无需构建步骤的单页 Dashboard。"""
        return HTMLResponse(_DASHBOARD_HTML.read_text(encoding="utf-8"))

    @router.get("/api/v1/dashboard/snapshot")
    def dashboard_snapshot() -> dict[str, Any]:
        """返回页面展示所需的确定性演示快照。"""
        return build_dashboard_snapshot(service, demo_run_id)

    return router

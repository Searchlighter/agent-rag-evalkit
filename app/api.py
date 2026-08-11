"""EvalKit HTTP API：管理数据集、评测、Trace、Badcase和回归比较。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .adapter_contract import MockRagAdapter
from .domain import BadcaseSeverity, BadcaseStatus, EvalRunStatus
from .service import EvalKitService


class CreateDatasetRequest(BaseModel):
    """创建逻辑评测集所需字段。"""
    name: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)


class ImportDatasetRequest(BaseModel):
    """导入一个不可变数据集版本的样本列表。"""
    cases: list[dict[str, Any]]
    actor_id: str = "api-user"


class CreateConfigRequest(BaseModel):
    """创建检索评测配置。"""
    name: str
    retrieval_k: int = Field(default=5, gt=0)


class CreateEvalRunRequest(BaseModel):
    """创建评测运行并绑定版本、配置和 Adapter。"""
    dataset_version_id: str
    config_id: str
    adapter_id: str
    budget_limit: float | None = Field(default=None, ge=0)


class ChangeRunStatusRequest(BaseModel):
    """请求合法的评测状态转换。"""
    status: EvalRunStatus


class CreateBadcaseRequest(BaseModel):
    """从运行结果创建 Badcase。"""
    case_id: str
    category: str
    note: str
    severity: BadcaseSeverity = BadcaseSeverity.MEDIUM


class UpdateBadcaseRequest(BaseModel):
    """记录 Badcase 分诊和处置结果。"""
    status: BadcaseStatus
    root_cause: str = ""
    owner: str = ""
    resolution: str = ""
    responsible_version: str = ""
    manual_conclusion: str = ""
    actor: str = "api-user"


class CreateRegressionSuiteRequest(BaseModel):
    """从指定 Badcase 创建回归集。"""
    name: str
    badcase_ids: list[str]


class ReviewCaseResultRequest(BaseModel):
    """保存单条结果的人工评分。"""
    score: float = Field(ge=0, le=5)
    reviewer: str = Field(min_length=1)
    note: str = ""


def create_router(service: EvalKitService) -> APIRouter:
    """将应用服务绑定为版本化 REST 路由。"""
    router = APIRouter(prefix="/api/v1")

    @router.post("/datasets")
    def create_dataset(request: CreateDatasetRequest) -> dict[str, Any]:
        """创建数据集。"""
        try:
            return asdict(service.create_dataset(request.name, request.owner_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/datasets/{dataset_id}/versions")
    def import_dataset_version(dataset_id: str, request: ImportDatasetRequest) -> dict[str, Any]:
        """校验样本并导入新的数据集版本。"""
        try:
            version = service.import_dataset_version(dataset_id, request.cases, request.actor_id)
            return {
                "id": version.id,
                "dataset_id": version.dataset_id,
                "version_number": version.version_number,
                "checksum": version.checksum,
                "case_count": len(version.cases),
            }
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/configs")
    def create_config(request: CreateConfigRequest) -> dict[str, Any]:
        """创建评测配置。"""
        try:
            return asdict(service.create_config(request.name, request.retrieval_k))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/eval-runs")
    def create_eval_run(request: CreateEvalRunRequest) -> dict[str, Any]:
        """创建 queued 状态的评测运行。"""
        try:
            return asdict(service.create_eval_run(**request.model_dump()))
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/eval-runs/{run_id}/status")
    def update_status(run_id: str, request: ChangeRunStatusRequest) -> dict[str, Any]:
        """更新评测运行状态。"""
        try:
            return asdict(service.update_eval_run_status(run_id, request.status))
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """查询运行摘要和聚合指标。"""
        try:
            return service.get_run_summary(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/eval-runs/{run_id}/execute")
    def execute_run(run_id: str) -> dict[str, Any]:
        """通过内置 Mock Adapter 执行同步评测。"""
        try:
            return service.execute_eval_run(run_id, adapter=MockRagAdapter())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/results")
    def list_results(run_id: str) -> list[dict[str, Any]]:
        """列出运行中的逐样本结果。"""
        try:
            service.get_run_summary(run_id)
            return [asdict(item) for item in service.list_case_results(run_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/results.csv", response_class=PlainTextResponse)
    def export_results(run_id: str) -> str:
        """将确定性指标导出为 CSV。"""
        try:
            return service.export_case_results_csv(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/eval-runs/{run_id}/cases/{case_id}/review")
    def review_case_result(
        run_id: str, case_id: str, request: ReviewCaseResultRequest
    ) -> dict[str, Any]:
        """提交人工评分与备注。"""
        try:
            return asdict(
                service.review_case_result(run_id, case_id, **request.model_dump())
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/cases/{case_id}/trace")
    def get_case_trace(run_id: str, case_id: str) -> list[dict[str, Any]]:
        """查询指定样本的标准化 Trace。"""
        try:
            return [asdict(item) for item in service.get_case_trace(run_id, case_id)]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/cases/{case_id}/diagnosis")
    def diagnose_case(run_id: str, case_id: str) -> dict[str, Any]:
        """返回指定样本的确定性诊断。"""
        try:
            return service.diagnose_case(run_id, case_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/eval-runs/{run_id}/badcases")
    def create_badcase(run_id: str, request: CreateBadcaseRequest) -> dict[str, Any]:
        """将失败样本加入 Badcase 工作台。"""
        try:
            return asdict(service.create_badcase(run_id, **request.model_dump()))
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.patch("/badcases/{badcase_id}")
    def update_badcase(badcase_id: str, request: UpdateBadcaseRequest) -> dict[str, Any]:
        """更新 Badcase 生命周期和处置字段。"""
        try:
            return asdict(service.update_badcase(badcase_id, **request.model_dump()))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/badcases")
    def list_badcases(
        eval_run_id: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        owner: str | None = None,
    ) -> list[dict[str, Any]]:
        """按运行、分类、等级、状态或负责人筛选 Badcase。"""
        return [
            asdict(item)
            for item in service.list_badcases(
                eval_run_id=eval_run_id,
                category=category,
                severity=severity,
                status=status,
                owner=owner,
            )
        ]

    @router.get("/badcases/{badcase_id}")
    def get_badcase_detail(badcase_id: str) -> dict[str, Any]:
        """查询 Badcase 及其完整活动历史。"""
        try:
            return service.get_badcase_detail(badcase_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/regression-suites")
    def create_regression_suite(request: CreateRegressionSuiteRequest) -> dict[str, Any]:
        """创建固定样本范围的回归集。"""
        try:
            return asdict(service.create_regression_suite_from_badcases(**request.model_dump()))
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.get("/regression-suites/{suite_id}/compare")
    def compare_regression_runs(
        suite_id: str, baseline_run_id: str, candidate_run_id: str
    ) -> dict[str, Any]:
        """比较基线和候选运行。"""
        try:
            return service.compare_regression_runs(suite_id, baseline_run_id, candidate_run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/regression-comparisons")
    def list_regression_comparisons(suite_id: str | None = None) -> list[dict[str, Any]]:
        """列出全部或指定回归集的比较记录。"""
        return service.list_regression_comparisons(suite_id)

    @router.get("/regression-comparisons/{comparison_id}")
    def get_regression_comparison(comparison_id: str) -> dict[str, Any]:
        """查询一次持久化回归比较。"""
        try:
            return service.get_regression_comparison(comparison_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/audit-events")
    def list_audits() -> list[dict[str, Any]]:
        """列出当前进程中的审计事件。"""
        return [asdict(item) for item in service.list_audit_events()]

    return router

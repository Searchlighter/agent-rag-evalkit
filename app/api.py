"""EvalKit HTTP API：管理数据集、评测、Trace、Badcase和回归比较。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .adapter_registry import (
    AdapterDisabledError,
    AdapterNotFoundError,
    AdapterRegistry,
    create_default_registry,
)
from .domain import AdapterType, BadcaseSeverity, BadcaseStatus, EvalRunStatus, new_id
from .ingestion import DatasetUploadValidationError, parse_dataset_content
from .service import EvalKitService


class CreateDatasetRequest(BaseModel):
    """创建逻辑评测集所需字段。"""
    name: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)


class ImportDatasetRequest(BaseModel):
    """导入一个不可变数据集版本的样本列表。"""
    cases: list[dict[str, Any]]
    actor_id: str = "api-user"


class UploadDatasetRequest(BaseModel):
    """浏览器上传的 UTF-8 JSONL/CSV 文件及数据集元数据。"""

    name: str = Field(min_length=1, max_length=100)
    owner_id: str = Field(default="local-user", min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(max_length=5_000_000)
    actor_id: str = Field(default="local-user", min_length=1, max_length=100)


class CreateConfigRequest(BaseModel):
    """创建检索评测配置。"""
    name: str
    retrieval_k: int = Field(default=5, gt=0)


class CreateEvalRunRequest(BaseModel):
    """创建评测运行并绑定版本、配置和 Adapter。"""
    dataset_version_id: str
    config_id: str
    adapter_id: str
    adapter_version_id: str = ""
    adapter_snapshot: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    model_id: str = ""
    budget_limit: float | None = Field(default=None, ge=0)


class TestAdapterRequest(BaseModel):
    """对已注册 Adapter 发起一次不保存结果的轻量探测。"""

    adapter_id: str = Field(min_length=1)
    question: str = Field(default="EvalKit connection test", min_length=1, max_length=500)


class CreateAdapterRequest(BaseModel):
    """创建逻辑 Adapter 配置；连接参数在后续版本中单独维护。"""

    adapter_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    name: str = Field(min_length=1, max_length=100)
    adapter_type: AdapterType = AdapterType.HTTP


class UpdateAdapterRequest(BaseModel):
    """编辑 Adapter 名称或启停状态。"""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    enabled: bool | None = None


class ConfigureHttpAdapterRequest(BaseModel):
    """保存通用 HTTP Adapter 的非敏感连接参数。"""

    endpoint: str = Field(min_length=1, max_length=2048)
    auth_method: str = Field(default="none", pattern=r"^(none|bearer_env)$")
    bearer_token_env: str | None = Field(default=None, max_length=128)
    timeout_seconds: float = Field(default=15, gt=0, le=300)
    retries: int = Field(default=1, ge=0, le=5)


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


def create_router(
    service: EvalKitService, adapter_registry: AdapterRegistry | None = None
) -> APIRouter:
    """将应用服务绑定为版本化 REST 路由。"""
    router = APIRouter(prefix="/api/v1")
    adapters = adapter_registry or create_default_registry()

    @router.get("/adapters")
    def list_adapters() -> list[dict[str, object]]:
        """列出内置与用户创建的 Adapter，不返回任何密钥。"""
        return adapters.list_configs()

    @router.post("/adapters", status_code=201)
    def create_adapter(request: CreateAdapterRequest) -> dict[str, object]:
        """创建待配置的逻辑 Adapter。"""
        try:
            return adapters.create_config(
                request.adapter_id, request.name, request.adapter_type
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.patch("/adapters/{adapter_id}")
    def update_adapter(
        adapter_id: str, request: UpdateAdapterRequest
    ) -> dict[str, object]:
        """更新 Adapter 名称或启停状态。"""
        if request.name is None and request.enabled is None:
            raise HTTPException(status_code=400, detail="至少提供一个需要更新的字段")
        try:
            return adapters.update_config(
                adapter_id, name=request.name, enabled=request.enabled
            )
        except AdapterNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.delete("/adapters/{adapter_id}")
    def delete_adapter(adapter_id: str) -> dict[str, object]:
        """删除 Adapter 配置以及可能存在的运行时绑定。"""
        try:
            deleted = adapters.delete_config(adapter_id)
        except AdapterNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "deleted", "adapter": deleted}

    @router.put("/adapters/{adapter_id}/http-configuration")
    def configure_http_adapter(
        adapter_id: str, request: ConfigureHttpAdapterRequest
    ) -> dict[str, object]:
        """校验并保存 HTTP 连接参数，不接收或回显 Token 明文。"""
        try:
            return adapters.configure_http(
                adapter_id,
                endpoint=request.endpoint,
                timeout_seconds=request.timeout_seconds,
                retries=request.retries,
                auth_method=request.auth_method,
                bearer_token_env=request.bearer_token_env,
            )
        except AdapterNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/adapters/test")
    def test_adapter_connection(request: TestAdapterRequest) -> dict[str, Any]:
        """验证 Adapter 可调用，并返回不含回答正文的契约摘要。"""
        request_id = new_id("probe")
        try:
            adapter = adapters.resolve(request.adapter_id)
            response = adapter.invoke(request.question, request_id=request_id)
        except (AdapterNotFoundError, AdapterDisabledError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "adapter_connection_failed",
                    "error_type": type(error).__name__,
                },
            ) from error
        return {
            "status": "ok",
            "adapter_id": request.adapter_id,
            "request_id": request_id,
            "response_request_id_matches": response.request_id == request_id,
            "answer_present": bool(response.answer.strip()),
            "citation_count": len(response.citations),
            "retrieval_count": len(response.retrievals),
            "event_count": len(response.events),
        }

    @router.post("/datasets")
    def create_dataset(request: CreateDatasetRequest) -> dict[str, Any]:
        """创建数据集。"""
        try:
            return asdict(service.create_dataset(request.name, request.owner_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.get("/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        """列出当前进程内的数据集及其版本摘要。"""
        return service.list_datasets()

    @router.post("/datasets/upload", status_code=201)
    def upload_dataset(request: UploadDatasetRequest) -> dict[str, Any]:
        """预先校验上传内容，通过后创建逻辑数据集及首个版本。"""
        try:
            raw_cases = parse_dataset_content(request.filename, request.content)
            service.validate_dataset_cases(raw_cases)
            dataset = service.create_dataset(request.name, request.owner_id)
            version = service.import_dataset_version(
                dataset.id, raw_cases, request.actor_id
            )
        except DatasetUploadValidationError as error:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "dataset_validation_failed",
                    "issues": error.issues,
                },
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "dataset": asdict(dataset),
            "version": {
                "id": version.id,
                "version_number": version.version_number,
                "case_count": len(version.cases),
                "checksum": version.checksum,
            },
            "source": {"filename": request.filename},
        }

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

    @router.get("/datasets/{dataset_id}/versions")
    def list_dataset_versions(dataset_id: str) -> list[dict[str, Any]]:
        """列出指定数据集的全部不可变版本。"""
        try:
            return service.list_dataset_versions(dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/datasets/{dataset_id}/versions/{version_id}/cases")
    def preview_dataset_version(
        dataset_id: str, version_id: str, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """分页预览指定版本的评测样本。"""
        try:
            return service.preview_dataset_version(
                dataset_id, version_id, offset=offset, limit=limit
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
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
            payload = request.model_dump()
            binding = adapters.get_run_binding(request.adapter_id)
            if not payload["adapter_version_id"]:
                payload["adapter_version_id"] = binding["adapter_version_id"]
            if not payload["adapter_snapshot"]:
                payload["adapter_snapshot"] = binding["adapter_snapshot"]
            return asdict(service.create_eval_run(**payload))
        except (AdapterNotFoundError, AdapterDisabledError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.get("/eval-runs")
    def list_eval_runs() -> list[dict[str, Any]]:
        """列出当前进程内的全部评测运行及其汇总指标。"""
        return service.list_eval_runs()

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
        """使用运行绑定的 Adapter 执行同步评测。"""
        try:
            run = service.get_run_summary(run_id)
            adapter = adapters.resolve(run["adapter_id"])
            return service.execute_eval_run(run_id, adapter=adapter)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (AdapterNotFoundError, AdapterDisabledError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/results")
    def list_results(run_id: str) -> list[dict[str, Any]]:
        """列出运行中的逐样本结果。"""
        try:
            return service.list_case_result_details(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/results.csv", response_class=PlainTextResponse)
    def export_results(run_id: str) -> PlainTextResponse:
        """将确定性指标导出为 CSV。"""
        try:
            content = "\ufeff" + service.export_case_results_csv(run_id)
            return PlainTextResponse(
                content,
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="eval-results-{run_id}.csv"'
                    )
                },
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/eval-runs/{run_id}/results.json", response_class=JSONResponse)
    def export_results_json(run_id: str) -> JSONResponse:
        """将完整逐样本结果导出为 JSON 附件。"""
        try:
            content = service.list_case_result_details(run_id)
            return JSONResponse(
                jsonable_encoder(content),
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="eval-results-{run_id}.json"'
                    )
                },
            )
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

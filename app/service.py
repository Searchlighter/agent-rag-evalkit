"""EvalKit 应用服务：编排评测、诊断、人工复核、Badcase和回归比较。"""

from __future__ import annotations

import hashlib
import json
import csv
from dataclasses import asdict
from io import StringIO
from typing import Any, Iterable

from .domain import (
    AuditEvent,
    Badcase,
    BadcaseActivity,
    BadcaseSeverity,
    BadcaseStatus,
    CaseResult,
    Dataset,
    DatasetVersion,
    EvaluationConfig,
    EvalCase,
    EvalRun,
    EvalRunStatus,
    RegressionCaseResult,
    RegressionComparison,
    RegressionOutcome,
    RegressionSuite,
    TraceEvent,
    TraceEventType,
    new_id,
    now,
    validate_adapter_settings,
)
from .adapter_contract import TargetAgentAdapter
from .metrics import citation_coverage, empty_retrieval_rate, mean_metric, recall_at_k, reciprocal_rank
from .quality import validate_answer
from .repository import InMemoryRepository
from .trace import normalize_trace_events


class EvalKitService:
    """在领域模型、仓储和外部 Adapter 之间实现应用用例。"""

    def __init__(self, repository: InMemoryRepository) -> None:
        self.repository = repository

    # 数据集、配置与运行生命周期
    def create_dataset(self, name: str, owner_id: str) -> Dataset:
        """创建一个可继续导入版本的逻辑数据集。"""
        if not name.strip() or not owner_id.strip():
            raise ValueError("数据集名称和 owner_id 不能为空")
        dataset = self.repository.save_dataset(
            Dataset(id=new_id("dataset"), name=name.strip(), owner_id=owner_id.strip())
        )
        self._audit("create", "dataset", dataset.id, owner_id, {"name": dataset.name})
        return dataset

    def import_dataset_version(
        self, dataset_id: str, raw_cases: Iterable[dict[str, Any]], actor_id: str = "system"
    ) -> DatasetVersion:
        """校验样本并创建带 SHA-256 校验和的新版本快照。"""
        dataset = self._get_dataset(dataset_id)
        cases = [EvalCase.from_dict(raw) for raw in raw_cases]
        self._validate_cases(cases)
        canonical = json.dumps([asdict(case) for case in cases], ensure_ascii=False, sort_keys=True)
        checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        version_number = 1 + sum(
            version.dataset_id == dataset_id for version in self.repository.dataset_versions.values()
        )
        version = self.repository.save_dataset_version(
            DatasetVersion(
                id=new_id("dsv"),
                dataset_id=dataset_id,
                version_number=version_number,
                checksum=checksum,
                cases=cases,
            )
        )
        dataset.current_version_id = version.id
        self.repository.save_dataset(dataset)
        self._audit("import", "dataset_version", version.id, actor_id, {"case_count": len(cases)})
        return version

    def create_config(self, name: str, retrieval_k: int = 5) -> EvaluationConfig:
        """创建确定性检索评测配置。"""
        if not name.strip() or retrieval_k <= 0:
            raise ValueError("配置名称不能为空，retrieval_k 必须大于 0")
        return self.repository.save_config(
            EvaluationConfig(id=new_id("config"), name=name.strip(), retrieval_k=retrieval_k)
        )

    def create_eval_run(
        self,
        dataset_version_id: str,
        config_id: str,
        adapter_id: str,
        budget_limit: float | None = None,
        adapter_version_id: str = "",
        adapter_snapshot: dict[str, str | int | float | bool | None] | None = None,
        model_id: str = "",
    ) -> EvalRun:
        """绑定数据、配置及 Adapter 快照，创建可复现的 queued 运行。"""
        if dataset_version_id not in self.repository.dataset_versions:
            raise KeyError(f"数据集版本不存在：{dataset_version_id}")
        if config_id not in self.repository.configs:
            raise KeyError(f"评测配置不存在：{config_id}")
        if not adapter_id.strip() or (budget_limit is not None and budget_limit < 0):
            raise ValueError("adapter_id 不能为空且预算不能小于 0")
        snapshot = validate_adapter_settings(adapter_snapshot or {})
        run = self.repository.save_eval_run(
            EvalRun(
                id=new_id("run"),
                dataset_version_id=dataset_version_id,
                config_id=config_id,
                adapter_id=adapter_id.strip(),
                adapter_version_id=adapter_version_id.strip(),
                adapter_snapshot=snapshot,
                model_id=model_id.strip(),
                budget_limit=budget_limit,
            )
        )
        self._audit(
            "create",
            "eval_run",
            run.id,
            "system",
            {
                "adapter_id": run.adapter_id,
                "adapter_version_id": run.adapter_version_id,
                "model_id": run.model_id,
            },
        )
        return run

    def update_eval_run_status(self, run_id: str, status: EvalRunStatus) -> EvalRun:
        """通过领域状态机更新评测运行状态。"""
        run = self._get_run(run_id)
        run.transition_to(status)
        self._audit("transition", "eval_run", run.id, "system", {"status": status.value})
        return self.repository.save_eval_run(run)

    # Badcase 分诊与处理历史
    def create_badcase(
        self, eval_run_id: str, case_id: str, category: str, note: str, severity: str = "medium"
    ) -> Badcase:
        """为已评测样本创建可跟踪的失败记录。"""
        run = self._get_run(eval_run_id)
        allowed_categories = {
            "no_retrieval", "wrong_retrieval", "rerank_error", "context_insufficient",
            "hallucination", "tool_failure", "workflow_failure", "needs_manual_review",
        }
        if category not in allowed_categories:
            raise ValueError("Unsupported Badcase category")
        version = self.repository.dataset_versions[run.dataset_version_id]
        if case_id not in {case.id for case in version.cases}:
            raise KeyError(f"样本不属于当前评测任务：{case_id}")
        if not category.strip() or not note.strip():
            raise ValueError("category 与 note 不能为空")
        try:
            normalized_severity = BadcaseSeverity(severity)
        except ValueError as error:
            raise ValueError("severity must be low, medium, high, or critical") from error
        badcase = self.repository.save_badcase(
            Badcase(
                id=new_id("bad"), eval_run_id=eval_run_id, case_id=case_id,
                category=category.strip(), note=note.strip(), severity=normalized_severity,
            )
        )
        self.repository.append_badcase_activity(
            BadcaseActivity(
                id=new_id("activity"),
                badcase_id=badcase.id,
                action="created",
                actor="system",
                to_status=BadcaseStatus.OPEN,
                note=note.strip(),
            )
        )
        self._audit("create", "badcase", badcase.id, "system", {"category": category})
        return badcase

    def update_badcase(
        self, badcase_id: str, status: BadcaseStatus, root_cause: str = "",
        owner: str = "", resolution: str = "", responsible_version: str = "",
        manual_conclusion: str = "", actor: str = "system",
    ) -> Badcase:
        """更新责任、根因和结论，并追加不可变活动记录。"""
        if badcase_id not in self.repository.badcases:
            raise KeyError(f"Badcase not found: {badcase_id}")
        badcase = self.repository.badcases[badcase_id]
        if status == BadcaseStatus.FIXED and not (resolution.strip() or badcase.resolution):
            raise ValueError("resolution is required before marking a Badcase fixed")
        previous_status = badcase.status
        badcase.transition_to(status)
        badcase.root_cause = root_cause.strip() or badcase.root_cause
        badcase.owner = owner.strip() or badcase.owner
        badcase.responsible_version = responsible_version.strip() or badcase.responsible_version
        badcase.manual_conclusion = manual_conclusion.strip() or badcase.manual_conclusion
        badcase.resolution = resolution.strip() or badcase.resolution
        self.repository.append_badcase_activity(
            BadcaseActivity(
                id=new_id("activity"),
                badcase_id=badcase_id,
                action="status_changed",
                actor=actor.strip() or owner.strip() or "system",
                from_status=previous_status,
                to_status=status,
                note=manual_conclusion.strip() or resolution.strip(),
            )
        )
        self._audit("update", "badcase", badcase_id, actor or owner or "system", {"status": status.value})
        return self.repository.save_badcase(badcase)

    def list_badcases(
        self,
        *,
        eval_run_id: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        owner: str | None = None,
    ) -> list[Badcase]:
        """组合可选条件筛选 Badcase，并按更新时间倒序返回。"""
        records = list(self.repository.badcases.values())
        if eval_run_id:
            records = [item for item in records if item.eval_run_id == eval_run_id]
        if category:
            records = [item for item in records if item.category == category]
        if severity:
            records = [item for item in records if item.severity.value == severity]
        if status:
            records = [item for item in records if item.status.value == status]
        if owner:
            records = [item for item in records if item.owner == owner]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def get_badcase_detail(self, badcase_id: str) -> dict[str, Any]:
        """返回 Badcase 当前字段和完整活动历史。"""
        if badcase_id not in self.repository.badcases:
            raise KeyError(f"Badcase not found: {badcase_id}")
        return {
            "badcase": asdict(self.repository.badcases[badcase_id]),
            "activities": [
                asdict(item) for item in self.repository.list_badcase_activities(badcase_id)
            ],
        }

    # 回归集与基线/候选比较
    def create_regression_suite_from_badcases(self, name: str, badcase_ids: list[str]) -> RegressionSuite:
        """从历史 Badcase 固化样本范围和基线运行引用。"""
        if not name.strip() or not badcase_ids:
            raise ValueError("suite name and badcase ids are required")
        try:
            badcases = [self.repository.badcases[item] for item in badcase_ids]
        except KeyError as error:
            raise KeyError(f"Badcase not found: {error.args[0]}") from error
        suite = RegressionSuite(
            id=new_id("suite"),
            name=name.strip(),
            case_ids=sorted({item.case_id for item in badcases}),
            source_badcase_ids=badcase_ids,
            baseline_run_ids=sorted({item.eval_run_id for item in badcases}),
        )
        self._audit("create", "regression_suite", suite.id, "system", {"case_count": len(suite.case_ids)})
        return self.repository.save_regression_suite(suite)

    def compare_regression_runs(
        self, suite_id: str, baseline_run_id: str, candidate_run_id: str
    ) -> dict[str, Any]:
        """使用确定性指标比较基线与候选，并保留不可评测状态。"""
        if suite_id not in self.repository.regression_suites:
            raise KeyError(f"Regression suite not found: {suite_id}")
        suite = self.repository.regression_suites[suite_id]
        baseline_run = self._get_run(baseline_run_id)
        candidate_run = self._get_run(candidate_run_id)
        baseline = {item.case_id: item for item in self.list_case_results(baseline_run_id)}
        candidate = {item.case_id: item for item in self.list_case_results(candidate_run_id)}
        results: list[RegressionCaseResult] = []
        for case_id in suite.case_ids:
            before, after = baseline.get(case_id), candidate.get(case_id)
            if not before or not after:
                outcome = RegressionOutcome.NOT_EVALUABLE
                reason = "case result missing from baseline or candidate run"
                before_score = after_score = None
            elif after.status != "succeeded":
                outcome = RegressionOutcome.NOT_EVALUABLE
                reason = "candidate execution did not succeed"
                before_score = self._regression_score(before)
                after_score = self._regression_score(after)
            elif any(value is False for value in after.quality_checks.values() if isinstance(value, bool)):
                outcome = RegressionOutcome.NEEDS_MANUAL_REVIEW
                reason = "candidate failed a deterministic answer-quality check"
                before_score = self._regression_score(before)
                after_score = self._regression_score(after)
            else:
                before_score = self._regression_score(before)
                after_score = self._regression_score(after)
                if before_score is None or after_score is None:
                    outcome = RegressionOutcome.NEEDS_MANUAL_REVIEW
                    reason = "no comparable deterministic metric"
                elif after_score < before_score:
                    outcome = RegressionOutcome.REGRESSED
                    reason = "candidate deterministic score decreased"
                else:
                    outcome = RegressionOutcome.PASSED
                    reason = "candidate score is equal to or better than baseline"
            results.append(
                RegressionCaseResult(
                    case_id=case_id,
                    outcome=outcome,
                    baseline_score=before_score,
                    candidate_score=after_score,
                    reason=reason,
                )
            )
        comparison_record = self.repository.save_regression_comparison(
            RegressionComparison(
                id=new_id("comparison"),
                suite_id=suite_id,
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
                baseline_dataset_version_id=baseline_run.dataset_version_id,
                candidate_dataset_version_id=candidate_run.dataset_version_id,
                results=results,
            )
        )
        summary = {
            outcome.value: sum(item.outcome == outcome for item in results)
            for outcome in RegressionOutcome
        }
        self._audit(
            "compare",
            "regression_suite",
            suite_id,
            "system",
            {"comparison_id": comparison_record.id, **summary},
        )
        return {
            "comparison_id": comparison_record.id,
            "suite_id": suite_id,
            "summary": summary,
            "comparison": {item.case_id: item.outcome.value for item in results},
        }

    def get_regression_comparison(self, comparison_id: str) -> dict[str, Any]:
        """查询指定回归比较的完整结构。"""
        if comparison_id not in self.repository.regression_comparisons:
            raise KeyError(f"Regression comparison not found: {comparison_id}")
        return asdict(self.repository.regression_comparisons[comparison_id])

    def list_regression_comparisons(self, suite_id: str | None = None) -> list[dict[str, Any]]:
        """列出全部或指定回归集的比较记录。"""
        return [
            asdict(item)
            for item in self.repository.list_regression_comparisons(suite_id)
        ]

    # 面向 MCP/API 的筛选与聚合查询
    def get_badcase_summary(
        self,
        eval_run_id: str | None = None,
        severity: str | None = None,
        dataset_version_id: str | None = None,
        tags: list[str] | None = None,
        tag_match: str = "any",
        category: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """按运行、版本、标签、类别、状态和严重级别聚合 Badcase。"""
        if tag_match not in {"any", "all"}:
            raise ValueError("tag_match must be any or all")
        requested_tags = {item.strip() for item in (tags or []) if item.strip()}
        records: list[tuple[Badcase, DatasetVersion, EvalCase]] = []
        for item in self.repository.badcases.values():
            run = self._get_run(item.eval_run_id)
            version = self.repository.dataset_versions[run.dataset_version_id]
            case = next(case for case in version.cases if case.id == item.case_id)
            records.append((item, version, case))
        if eval_run_id:
            self._get_run(eval_run_id)
            records = [row for row in records if row[0].eval_run_id == eval_run_id]
        if dataset_version_id:
            if dataset_version_id not in self.repository.dataset_versions:
                raise KeyError(f"Dataset version not found: {dataset_version_id}")
            records = [row for row in records if row[1].id == dataset_version_id]
        if severity:
            records = [row for row in records if row[0].severity.value == severity]
        if category:
            records = [row for row in records if row[0].category == category]
        if status:
            records = [row for row in records if row[0].status.value == status]
        if requested_tags:
            records = [
                row
                for row in records
                if (
                    requested_tags.issubset(set(row[2].tags))
                    if tag_match == "all"
                    else bool(requested_tags.intersection(row[2].tags))
                )
            ]
        by_category: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_dataset_version: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        for item, version, case in records:
            by_category[item.category] = by_category.get(item.category, 0) + 1
            by_status[item.status.value] = by_status.get(item.status.value, 0) + 1
            by_severity[item.severity.value] = by_severity.get(item.severity.value, 0) + 1
            by_dataset_version[version.id] = by_dataset_version.get(version.id, 0) + 1
            for tag in case.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
        return {
            "count": len(records),
            "by_category": by_category,
            "by_status": by_status,
            "by_severity": by_severity,
            "by_dataset_version": by_dataset_version,
            "by_tag": by_tag,
            "filters": {
                "eval_run_id": eval_run_id,
                "dataset_version_id": dataset_version_id,
                "severity": severity,
                "category": category,
                "status": status,
                "tags": sorted(requested_tags),
                "tag_match": tag_match,
            },
            "items": [
                {
                    "id": item.id,
                    "eval_run_id": item.eval_run_id,
                    "dataset_version_id": version.id,
                    "case_id": item.case_id,
                    "tags": case.tags,
                    "category": item.category,
                    "severity": item.severity.value,
                    "status": item.status.value,
                }
                for item, version, case in records
            ],
        }

    def execute_eval_run(self, run_id: str, adapter: TargetAgentAdapter) -> dict[str, Any]:
        """同步执行数据集中的所有样本并保存结果与标准 Trace。"""
        run = self._get_run(run_id)
        if run.status != EvalRunStatus.QUEUED:
            raise ValueError(f"只有 queued 任务可执行，当前状态为：{run.status.value}")

        self.update_eval_run_status(run_id, EvalRunStatus.RUNNING)
        version = self.repository.dataset_versions[run.dataset_version_id]
        config = self.repository.configs[run.config_id]
        results: list[CaseResult] = []

        try:
            for case in version.cases:
                result = self._evaluate_case(run, case, config.retrieval_k, adapter)
                results.append(result)
                raw_events = result.trace_events or (
                    [{"type": "error", "name": "adapter", "error": result.error}]
                    if result.error else []
                )
                self.repository.replace_trace_events(
                    run_id,
                    case.id,
                    normalize_trace_events(raw_events, eval_run_id=run_id, case_id=case.id),
                )
            self.repository.replace_case_results(run_id, results)
            self.update_eval_run_status(run_id, EvalRunStatus.SUCCEEDED)
            self._audit("execute", "eval_run", run_id, "system", {"case_count": len(results)})
        except Exception as error:
            self.repository.replace_case_results(run_id, results)
            self.update_eval_run_status(run_id, EvalRunStatus.FAILED)
            self._audit("fail", "eval_run", run_id, "system", {"error": str(error)})
            raise

        return self.get_run_summary(run_id)

    def list_case_results(self, run_id: str) -> list[CaseResult]:
        """查询一次运行的全部样本结果。"""
        self._get_run(run_id)
        return self.repository.list_case_results(run_id)

    def export_case_results_csv(self, run_id: str) -> str:
        """将核心确定性指标输出为 UTF-8 CSV 文本。"""
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["case_id", "status", "recall_at_k", "mrr", "citation_coverage", "error"],
        )
        writer.writeheader()
        for item in self.list_case_results(run_id):
            writer.writerow({
                "case_id": item.case_id,
                "status": item.status,
                "recall_at_k": item.metrics.get("recall_at_k"),
                "mrr": item.metrics.get("mrr"),
                "citation_coverage": item.metrics.get("citation_coverage"),
                "error": item.error or "",
            })
        return output.getvalue()

    def review_case_result(
        self,
        run_id: str,
        case_id: str,
        score: float,
        reviewer: str,
        note: str = "",
    ) -> CaseResult:
        """保存0到5分的人工评分和复核说明。"""
        if not 0 <= score <= 5:
            raise ValueError("human score must be between 0 and 5")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        result = self._get_case_result(run_id, case_id)
        result.human_score = round(float(score), 2)
        result.human_note = note.strip()
        result.reviewed_by = reviewer.strip()
        result.reviewed_at = now()
        self.repository.save_case_result(result)
        self._audit(
            "review",
            "case_result",
            result.id,
            reviewer.strip(),
            {"score": result.human_score, "eval_run_id": run_id, "case_id": case_id},
        )
        return result

    def get_case_trace(self, run_id: str, case_id: str) -> list[TraceEvent]:
        """查询单条样本的顺序化 Trace。"""
        self._get_case(run_id, case_id)
        return self.repository.list_trace_events(run_id, case_id)

    def export_case_trace(
        self,
        run_id: str,
        case_id: str,
        exporter: Any,
        **export_options: Any,
    ) -> str:
        """通过可插拔 exporter 导出单条 Case Trace。"""
        result = self._get_case_result(run_id, case_id)
        events = self.get_case_trace(run_id, case_id)
        trace_id = exporter.export_case(result, events, **export_options)
        self._audit(
            "export",
            "langfuse_trace",
            trace_id,
            "system",
            {"eval_run_id": run_id, "case_id": case_id},
        )
        return trace_id

    def diagnose_case(self, run_id: str, case_id: str) -> dict[str, Any]:
        """依据执行状态、检索和质量检查给出确定性失败分类。"""
        case = self._get_case(run_id, case_id)
        result = self._get_case_result(run_id, case_id)
        trace = self.get_case_trace(run_id, case_id)
        if result.status == "failed":
            category, suggestion = "tool_failure", "检查 Adapter 契约、超时、重试与外部依赖。"
        elif case.expected_evidence and not result.retrieval_ids:
            category, suggestion = "no_retrieval", "检查索引、Query Rewrite 与过滤条件。"
        elif result.metrics.get("recall_at_k") == 0:
            category, suggestion = "wrong_retrieval", "检查 Embedding、关键词召回、元数据过滤和重排。"
        elif not result.quality_checks.get("citation_presence_pass", True):
            category, suggestion = "hallucination", "要求回答附引用，并在生成前验证证据存在。"
        elif not result.quality_checks.get("keyword_pass", True):
            category, suggestion = "context_insufficient", "补充证据、调整上下文预算或优化回答模板。"
        else:
            category, suggestion = "needs_manual_review", "确定性检查通过，需要人工复核语义正确性。"
        return {
            "case_id": case_id,
            "suggested_category": category,
            "suggestion": suggestion,
            "trace_event_count": len(trace),
            "quality_checks": result.quality_checks,
        }

    def explain_retrieval(self, run_id: str, case_id: str) -> dict[str, Any]:
        """返回完整候选、过滤标记、期望差异及相关检索 Trace。"""
        case = self._get_case(run_id, case_id)
        result = self._get_case_result(run_id, case_id)
        run = self._get_run(run_id)
        config = self.repository.configs[run.config_id]
        expected = set(case.expected_evidence)
        candidates: list[dict[str, Any]] = []
        for raw in sorted(
            result.retrieval_candidates,
            key=lambda item: int(item.get("rank", 10**9)),
        ):
            evidence_id = str(raw["evidence_id"])
            metadata = dict(raw.get("metadata", {}))
            filtered = bool(metadata.get("filtered", False))
            rank = int(raw["rank"])
            candidates.append({
                "evidence_id": evidence_id,
                "document_id": raw["document_id"],
                "chunk_id": raw["chunk_id"],
                "score": raw["score"],
                "rerank_score": metadata.get("rerank_score"),
                "rank": rank,
                "selected": not filtered and rank <= config.retrieval_k,
                "filtered": filtered,
                "filter_reason": metadata.get("filter_reason"),
                "expected_match": evidence_id in expected,
                "metadata": metadata,
            })
        retrieved = {item["evidence_id"] for item in candidates if not item["filtered"]}
        relevant_trace = [
            asdict(item)
            for item in self.get_case_trace(run_id, case_id)
            if item.event_type in {TraceEventType.RETRIEVAL, TraceEventType.RERANK}
        ]
        return {
            "eval_run_id": run_id,
            "case_id": case_id,
            "retrieval_k": config.retrieval_k,
            "expected_evidence": sorted(expected),
            "citations": result.citations,
            "candidates": candidates,
            "missing_expected_evidence": sorted(expected - retrieved),
            "unexpected_retrievals": sorted(retrieved - expected),
            "trace": relevant_trace,
            "diagnosis": self.diagnose_case(run_id, case_id),
        }

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """聚合运行状态、完成数、Badcase数和可计算指标。"""
        run = self._get_run(run_id)
        version = self.repository.dataset_versions[run.dataset_version_id]
        badcase_count = sum(item.eval_run_id == run_id for item in self.repository.badcases.values())
        results = self.repository.list_case_results(run_id)
        return {
            "run_id": run.id,
            "status": run.status.value,
            "dataset_version_id": version.id,
            "dataset_case_count": len(version.cases),
            "adapter_id": run.adapter_id,
            "adapter_version_id": run.adapter_version_id,
            "adapter_snapshot": dict(run.adapter_snapshot),
            "model_id": run.model_id,
            "badcase_count": badcase_count,
            "completed_case_count": len(results),
            "metrics": {
                "recall_at_k": mean_metric([item.metrics.get("recall_at_k") for item in results]),
                "mrr": mean_metric([item.metrics.get("mrr") for item in results]),
                "citation_coverage": mean_metric(
                    [item.metrics.get("citation_coverage") for item in results]
                ),
                "empty_retrieval_rate": empty_retrieval_rate(
                    [item.retrieval_ids for item in results if item.status == "succeeded"]
                ),
            },
            "note": "指标与规则校验均为确定性计算；语义正确性可通过人工评分补充。",
        }

    def list_audit_events(self) -> list[AuditEvent]:
        """返回当前仓储中的审计事件快照。"""
        return list(self.repository.audit_events)

    def record_external_audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        actor_id: str,
        details: dict[str, Any],
    ) -> None:
        """供 MCP 等外围接口写入统一审计事件。"""
        self._audit(action, resource_type, resource_id, actor_id, details)

    def _get_dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.repository.datasets:
            raise KeyError(f"数据集不存在：{dataset_id}")
        return self.repository.datasets[dataset_id]

    def _get_run(self, run_id: str) -> EvalRun:
        if run_id not in self.repository.eval_runs:
            raise KeyError(f"评测任务不存在：{run_id}")
        return self.repository.eval_runs[run_id]

    def _get_case(self, run_id: str, case_id: str) -> EvalCase:
        run = self._get_run(run_id)
        for case in self.repository.dataset_versions[run.dataset_version_id].cases:
            if case.id == case_id:
                return case
        raise KeyError(f"样本不属于当前评测任务：{case_id}")

    def _get_case_result(self, run_id: str, case_id: str) -> CaseResult:
        for item in self.repository.list_case_results(run_id):
            if item.case_id == case_id:
                return item
        raise KeyError(f"样本尚未产生评测结果：{case_id}")

    @staticmethod
    def _regression_score(result: CaseResult) -> float | None:
        values = [
            value
            for key in ("recall_at_k", "mrr", "citation_coverage")
            if (value := result.metrics.get(key)) is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    def _audit(self, action: str, resource_type: str, resource_id: str, actor_id: str, details: dict[str, Any]) -> None:
        self.repository.append_audit_event(
            AuditEvent(
                id=new_id("audit"), action=action, resource_type=resource_type,
                resource_id=resource_id, actor_id=actor_id, details=details,
            )
        )

    @staticmethod
    def _evaluate_case(
        run: EvalRun, case: EvalCase, retrieval_k: int, adapter: TargetAgentAdapter
    ) -> CaseResult:
        request_id = f"{run.id}:{case.id}"
        try:
            response = adapter.invoke(case.question, request_id=request_id)
            retrieval_ids = [f"{item.document_id}#{item.chunk_id}" for item in response.retrievals]
            retrieval_candidates = [
                {
                    "evidence_id": f"{item.document_id}#{item.chunk_id}",
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                    "rank": item.rank,
                    "metadata": item.metadata,
                }
                for item in response.retrievals
            ]
            quality_checks = validate_answer(
                response.answer,
                response.citations,
                case.metadata,
                retrieval_ids,
            )
            return CaseResult(
                id=new_id("result"),
                eval_run_id=run.id,
                case_id=case.id,
                status="succeeded",
                answer=response.answer,
                citations=response.citations,
                retrieval_ids=retrieval_ids,
                retrieval_candidates=retrieval_candidates,
                metrics={
                    "recall_at_k": recall_at_k(case.expected_evidence, retrieval_ids, retrieval_k),
                    "mrr": reciprocal_rank(case.expected_evidence, retrieval_ids),
                    "citation_coverage": citation_coverage(case.expected_evidence, response.citations),
                },
                quality_checks=quality_checks,
                trace_events=response.events,
            )
        except Exception as error:
            return CaseResult(
                id=new_id("result"),
                eval_run_id=run.id,
                case_id=case.id,
                status="failed",
                error=str(error),
            )

    @staticmethod
    def _validate_cases(cases: list[EvalCase]) -> None:
        if not cases:
            raise ValueError("至少需要导入一条评测样本")
        ids = [case.id for case in cases]
        duplicate_ids = {item for item in ids if ids.count(item) > 1}
        if duplicate_ids:
            raise ValueError(f"评测样本 id 重复：{sorted(duplicate_ids)}")

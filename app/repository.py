"""领域对象的进程内仓储实现，用于本地演示和测试。"""

from __future__ import annotations

from .domain import (
    AuditEvent,
    Badcase,
    BadcaseActivity,
    CaseResult,
    Dataset,
    DatasetVersion,
    EvaluationConfig,
    EvalRun,
    RegressionComparison,
    RegressionSuite,
    TraceEvent,
)


class InMemoryRepository:
    """以字典保存领域对象；进程重启后数据会丢失且不提供跨进程一致性。"""

    def __init__(self) -> None:
        # 主实体按ID索引，活动和审计事件保留追加顺序。
        self.datasets: dict[str, Dataset] = {}
        self.dataset_versions: dict[str, DatasetVersion] = {}
        self.configs: dict[str, EvaluationConfig] = {}
        self.eval_runs: dict[str, EvalRun] = {}
        self.case_results: dict[str, CaseResult] = {}
        self.badcases: dict[str, Badcase] = {}
        self.badcase_activities: list[BadcaseActivity] = []
        self.trace_events: dict[str, TraceEvent] = {}
        self.regression_suites: dict[str, RegressionSuite] = {}
        self.regression_comparisons: dict[str, RegressionComparison] = {}
        self.audit_events: list[AuditEvent] = []

    def save_dataset(self, item: Dataset) -> Dataset:
        """新增或覆盖数据集。"""
        self.datasets[item.id] = item
        return item

    def save_dataset_version(self, item: DatasetVersion) -> DatasetVersion:
        """保存数据集版本。"""
        self.dataset_versions[item.id] = item
        return item

    def save_config(self, item: EvaluationConfig) -> EvaluationConfig:
        """保存评测配置。"""
        self.configs[item.id] = item
        return item

    def save_eval_run(self, item: EvalRun) -> EvalRun:
        """保存评测运行。"""
        self.eval_runs[item.id] = item
        return item

    def save_case_result(self, item: CaseResult) -> CaseResult:
        """保存单条评测结果。"""
        self.case_results[item.id] = item
        return item

    def replace_case_results(self, eval_run_id: str, results: list[CaseResult]) -> None:
        """原子语义地替换某次运行的结果集合；当前内存实现不提供线程锁。"""
        for result_id, item in list(self.case_results.items()):
            if item.eval_run_id == eval_run_id:
                del self.case_results[result_id]
        for item in results:
            self.save_case_result(item)

    def list_case_results(self, eval_run_id: str) -> list[CaseResult]:
        """按运行ID返回全部样本结果。"""
        return [item for item in self.case_results.values() if item.eval_run_id == eval_run_id]

    def save_badcase(self, item: Badcase) -> Badcase:
        """保存 Badcase 当前状态。"""
        self.badcases[item.id] = item
        return item

    def append_badcase_activity(self, item: BadcaseActivity) -> BadcaseActivity:
        """追加 Badcase 活动记录。"""
        self.badcase_activities.append(item)
        return item

    def list_badcase_activities(self, badcase_id: str) -> list[BadcaseActivity]:
        """返回一个 Badcase 的处理历史。"""
        return [item for item in self.badcase_activities if item.badcase_id == badcase_id]

    def save_trace_event(self, item: TraceEvent) -> TraceEvent:
        """保存一条标准化 Trace。"""
        self.trace_events[item.id] = item
        return item

    def replace_trace_events(self, eval_run_id: str, case_id: str, events: list[TraceEvent]) -> None:
        """替换指定运行和样本的全部标准化事件。"""
        for event_id, item in list(self.trace_events.items()):
            if item.eval_run_id == eval_run_id and item.case_id == case_id:
                del self.trace_events[event_id]
        for item in events:
            self.save_trace_event(item)

    def list_trace_events(self, eval_run_id: str, case_id: str | None = None) -> list[TraceEvent]:
        """按运行和可选样本ID顺序返回 Trace。"""
        events = [
            item for item in self.trace_events.values()
            if item.eval_run_id == eval_run_id and (case_id is None or item.case_id == case_id)
        ]
        return sorted(events, key=lambda item: (item.case_id, item.sequence))

    def save_regression_suite(self, item: RegressionSuite) -> RegressionSuite:
        """保存回归集。"""
        self.regression_suites[item.id] = item
        return item

    def save_regression_comparison(self, item: RegressionComparison) -> RegressionComparison:
        """保存回归比较记录。"""
        self.regression_comparisons[item.id] = item
        return item

    def list_regression_comparisons(self, suite_id: str | None = None) -> list[RegressionComparison]:
        """按更新时间倒序返回回归比较。"""
        records = list(self.regression_comparisons.values())
        if suite_id:
            records = [item for item in records if item.suite_id == suite_id]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def append_audit_event(self, item: AuditEvent) -> AuditEvent:
        """追加资源操作审计事件。"""
        self.audit_events.append(item)
        return item

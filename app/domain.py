"""评测数据集、运行状态、Trace、Badcase和回归比较的领域模型。"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


def now() -> datetime:
    """返回带 UTC 时区的当前时间。"""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """生成便于排查日志的短前缀资源ID。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class EvalRunStatus(str, Enum):
    """评测运行的有限状态集合。"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


class AdapterType(str, Enum):
    """EvalKit 当前识别的被测系统接入类型。"""

    MOCK = "mock"
    HTTP = "http"
    DIFY = "dify"
    RAGFLOW = "ragflow"
    LANGGRAPH = "langgraph"


class TraceEventType(str, Enum):
    """跨框架统一后的执行事件类型。"""
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    LLM = "llm"
    TOOL = "tool"
    ERROR = "error"
    FINAL = "final"


class BadcaseStatus(str, Enum):
    """Badcase 从发现到关闭的处理状态。"""
    OPEN = "open"
    TRIAGED = "triaged"
    FIXED = "fixed"
    IGNORED = "ignored"


class BadcaseSeverity(str, Enum):
    """Badcase 对业务与评测可信度的影响级别。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RegressionOutcome(str, Enum):
    """候选运行相对基线的单样本比较结论。"""
    PASSED = "passed"
    REGRESSED = "regressed"
    NOT_EVALUABLE = "not_evaluable"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"


TERMINAL_STATUSES = {
    EvalRunStatus.SUCCEEDED,
    EvalRunStatus.FAILED,
    EvalRunStatus.CANCELLED,
    EvalRunStatus.BUDGET_EXCEEDED,
}

ALLOWED_TRANSITIONS = {
    EvalRunStatus.QUEUED: {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLED},
    EvalRunStatus.RUNNING: TERMINAL_STATUSES,
}


def _string_list(raw: object, field_name: str) -> list[str]:
    """严格校验 JSON 数组字段，避免字符串被错误拆成字符列表。"""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"EvalCase.{field_name} 必须是字符串数组")
    return list(raw)


@dataclass(slots=True)
class EvalCase:
    """一条问题、期望答案、证据和规则元数据组成的评测样本。"""
    id: str
    question: str
    expected_answers: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalCase":
        """从外部字典构造样本并严格校验复合字段类型。"""
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError("EvalCase.question 不能为空")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("EvalCase.metadata 必须是对象")
        return cls(
            id=str(raw.get("id") or new_id("case")),
            question=question,
            expected_answers=_string_list(raw.get("expected_answers"), "expected_answers"),
            expected_evidence=_string_list(raw.get("expected_evidence"), "expected_evidence"),
            tags=_string_list(raw.get("tags"), "tags"),
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class Dataset:
    """评测集逻辑容器及其当前版本引用。"""
    id: str
    name: str
    owner_id: str
    current_version_id: str | None = None
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class DatasetVersion:
    """带校验和的不可变评测样本快照。"""
    id: str
    dataset_id: str
    version_number: int
    checksum: str
    cases: list[EvalCase]
    schema_name: str = "agent_rag_eval_case/v1"
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class EvaluationConfig:
    """一次评测使用的检索与可复现参数。"""
    id: str
    name: str
    retrieval_k: int = 5
    timeout_seconds: int = 30
    max_concurrency: int = 1
    seed: int = 42


@dataclass(slots=True)
class AdapterConfig:
    """可启停的 Adapter 逻辑配置及当前版本引用。"""

    id: str
    name: str
    adapter_type: AdapterType
    current_version_id: str | None = None
    enabled: bool = True
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.name = self.name.strip()
        if not self.id or not self.name:
            raise ValueError("AdapterConfig.id 和 name 不能为空")
        if not isinstance(self.adapter_type, AdapterType):
            raise ValueError("AdapterConfig.adapter_type 必须是受支持的 AdapterType")


AdapterSettingValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class AdapterVersion:
    """不可变的 Adapter 运行配置快照，不保存密钥明文。"""

    id: str
    adapter_config_id: str
    version_number: int
    adapter_type: AdapterType
    settings: Mapping[str, AdapterSettingValue] = field(default_factory=dict)
    secret_refs: Mapping[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)

    def __post_init__(self) -> None:
        adapter_id = self.id.strip()
        config_id = self.adapter_config_id.strip()
        if not adapter_id or not config_id:
            raise ValueError("AdapterVersion.id 和 adapter_config_id 不能为空")
        if self.version_number < 1:
            raise ValueError("AdapterVersion.version_number 必须大于 0")
        if not isinstance(self.adapter_type, AdapterType):
            raise ValueError("AdapterVersion.adapter_type 必须是受支持的 AdapterType")

        settings = self._validated_settings(self.settings)
        secret_refs = self._validated_secret_refs(self.secret_refs)
        object.__setattr__(self, "id", adapter_id)
        object.__setattr__(self, "adapter_config_id", config_id)
        object.__setattr__(self, "settings", MappingProxyType(settings))
        object.__setattr__(self, "secret_refs", MappingProxyType(secret_refs))

    def public_snapshot(self) -> dict[str, Any]:
        """返回可用于 API、日志和审计的脱敏版本快照。"""
        return {
            "id": self.id,
            "adapter_config_id": self.adapter_config_id,
            "version_number": self.version_number,
            "adapter_type": self.adapter_type.value,
            "settings": dict(self.settings),
            "secret_refs": {name: "***" for name in self.secret_refs},
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def _validated_settings(
        raw: Mapping[str, AdapterSettingValue],
    ) -> dict[str, AdapterSettingValue]:
        settings: dict[str, AdapterSettingValue] = {}
        for key, value in raw.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                raise ValueError("AdapterVersion.settings 的键不能为空")
            if any(
                marker in normalized_key.lower()
                for marker in ("api_key", "password", "secret", "token")
            ):
                raise ValueError(
                    f"AdapterVersion.settings 禁止保存敏感字段: {normalized_key}"
                )
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"AdapterVersion.settings 不支持复杂值: {normalized_key}")
            settings[normalized_key] = value
        return settings

    @staticmethod
    def _validated_secret_refs(raw: Mapping[str, str]) -> dict[str, str]:
        secret_refs: dict[str, str] = {}
        for key, value in raw.items():
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if not normalized_key or not normalized_value:
                raise ValueError("AdapterVersion.secret_refs 的名称和引用不能为空")
            secret_refs[normalized_key] = normalized_value
        return secret_refs


@dataclass(slots=True)
class EvalRun:
    """把数据集版本、配置和目标 Adapter 绑定为一次运行。"""
    id: str
    dataset_version_id: str
    config_id: str
    adapter_id: str
    status: EvalRunStatus = EvalRunStatus.QUEUED
    budget_limit: float | None = None
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def transition_to(self, next_status: EvalRunStatus) -> None:
        """只允许状态机中声明的单向流转。"""
        if self.status in TERMINAL_STATUSES:
            raise ValueError(f"终态任务不能转换：{self.status} -> {next_status}")
        if next_status not in ALLOWED_TRANSITIONS.get(self.status, set()):
            raise ValueError(f"非法状态转换：{self.status} -> {next_status}")
        self.status = next_status
        self.updated_at = now()


@dataclass(slots=True)
class Badcase:
    """对失败样本的归类、责任和处置记录。"""
    id: str
    eval_run_id: str
    case_id: str
    category: str
    note: str
    severity: BadcaseSeverity = BadcaseSeverity.MEDIUM
    status: BadcaseStatus = BadcaseStatus.OPEN
    root_cause: str = ""
    owner: str = ""
    responsible_version: str = ""
    manual_conclusion: str = ""
    resolution: str = ""
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def transition_to(self, next_status: BadcaseStatus) -> None:
        """执行不可逆的 Badcase 状态流转。"""
        allowed = {
            BadcaseStatus.OPEN: {BadcaseStatus.TRIAGED, BadcaseStatus.IGNORED},
            BadcaseStatus.TRIAGED: {BadcaseStatus.FIXED, BadcaseStatus.IGNORED},
            BadcaseStatus.FIXED: set(),
            BadcaseStatus.IGNORED: set(),
        }
        if next_status not in allowed[self.status]:
            raise ValueError(f"非法 Badcase 状态流转：{self.status} -> {next_status}")
        self.status = next_status
        self.updated_at = now()


@dataclass(slots=True)
class BadcaseActivity:
    """Badcase 生命周期中的一条审计活动。"""
    id: str
    badcase_id: str
    action: str
    actor: str
    from_status: BadcaseStatus | None = None
    to_status: BadcaseStatus | None = None
    note: str = ""
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class CaseResult:
    """一次评测任务中单条样本的执行与指标结果。"""

    id: str
    eval_run_id: str
    case_id: str
    status: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    retrieval_ids: list[str] = field(default_factory=list)
    retrieval_candidates: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    quality_checks: dict[str, object] = field(default_factory=dict)
    human_score: float | None = None
    human_note: str = ""
    reviewed_by: str = ""
    reviewed_at: datetime | None = None
    error: str | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class TraceEvent:
    """关联运行和样本的标准化执行事件。"""
    id: str
    trace_id: str
    eval_run_id: str
    case_id: str
    event_type: TraceEventType
    node_name: str
    sequence: int
    source: str = "adapter"
    source_event_type: str = ""
    duration_ms: float | None = None
    input_summary: str = ""
    output_summary: str = ""
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class RegressionSuite:
    """由历史 Badcase 沉淀出的固定回归范围。"""
    id: str
    name: str
    case_ids: list[str]
    source_badcase_ids: list[str] = field(default_factory=list)
    baseline_run_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class RegressionCaseResult:
    """基线与候选运行在单个样本上的比较结果。"""
    case_id: str
    outcome: RegressionOutcome
    baseline_score: float | None = None
    candidate_score: float | None = None
    reason: str = ""


@dataclass(slots=True)
class RegressionComparison:
    """一次持久化的版本级回归比较。"""
    id: str
    suite_id: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_dataset_version_id: str
    candidate_dataset_version_id: str
    results: list[RegressionCaseResult]
    created_at: datetime = field(default_factory=now)


@dataclass(slots=True)
class AuditEvent:
    """记录关键资源操作的轻量审计事件。"""
    id: str
    action: str
    resource_type: str
    resource_id: str
    actor_id: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)

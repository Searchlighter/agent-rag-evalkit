"""运行时 Adapter 注册与解析。"""

from __future__ import annotations

from collections.abc import Mapping

from .adapter_contract import MockRagAdapter, TargetAgentAdapter


class AdapterNotFoundError(LookupError):
    """评测运行引用了当前应用未注册的 Adapter。"""


class AdapterRegistry:
    """使用稳定 adapter_id 管理被测 Agent/RAG Adapter。"""

    def __init__(self, adapters: Mapping[str, TargetAgentAdapter] | None = None) -> None:
        self._adapters: dict[str, TargetAgentAdapter] = {}
        for adapter_id, adapter in (adapters or {}).items():
            self.register(adapter_id, adapter)

    def register(
        self, adapter_id: str, adapter: TargetAgentAdapter, *, replace: bool = False
    ) -> None:
        """注册 Adapter；默认拒绝静默覆盖已有配置。"""
        normalized_id = adapter_id.strip()
        if not normalized_id:
            raise ValueError("adapter_id 不能为空")
        if not callable(getattr(adapter, "invoke", None)):
            raise TypeError("adapter 必须实现 invoke(question, request_id=...) 方法")
        if normalized_id in self._adapters and not replace:
            raise ValueError(f"Adapter 已注册: {normalized_id}")
        self._adapters[normalized_id] = adapter

    def resolve(self, adapter_id: str) -> TargetAgentAdapter:
        """按 ID 获取 Adapter，未知 ID 返回可诊断错误。"""
        normalized_id = adapter_id.strip()
        try:
            return self._adapters[normalized_id]
        except KeyError as error:
            raise AdapterNotFoundError(f"Adapter 未注册: {normalized_id}") from error

    def list_ids(self) -> tuple[str, ...]:
        """返回稳定排序的已注册 Adapter ID。"""
        return tuple(sorted(self._adapters))


def create_default_registry() -> AdapterRegistry:
    """创建兼容当前 Demo 的默认 Adapter 注册表。"""
    return AdapterRegistry({"mock-rag": MockRagAdapter()})

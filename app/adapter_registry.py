"""运行时 Adapter 注册与解析。"""

from __future__ import annotations

from collections.abc import Mapping

from .adapter_contract import MockRagAdapter, TargetAgentAdapter
from .domain import AdapterConfig, AdapterType, now


class AdapterNotFoundError(LookupError):
    """评测运行引用了当前应用未注册的 Adapter。"""


class AdapterDisabledError(LookupError):
    """评测运行引用了已停用的 Adapter。"""


class AdapterRegistry:
    """使用稳定 adapter_id 管理被测 Agent/RAG Adapter。"""

    def __init__(self, adapters: Mapping[str, TargetAgentAdapter] | None = None) -> None:
        self._adapters: dict[str, TargetAgentAdapter] = {}
        self._configs: dict[str, AdapterConfig] = {}
        self._system_ids: set[str] = set()
        for adapter_id, adapter in (adapters or {}).items():
            self.register(adapter_id, adapter)
            self._system_ids.add(adapter_id.strip())

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
        if normalized_id not in self._configs:
            adapter_type = (
                AdapterType.MOCK if isinstance(adapter, MockRagAdapter) else AdapterType.HTTP
            )
            self._configs[normalized_id] = AdapterConfig(
                normalized_id,
                "内置 Mock RAG" if adapter_type is AdapterType.MOCK else normalized_id,
                adapter_type,
            )

    def create_config(
        self, adapter_id: str, name: str, adapter_type: AdapterType
    ) -> dict[str, object]:
        """创建尚未绑定运行时实现的逻辑 Adapter。"""
        config = AdapterConfig(adapter_id, name, adapter_type)
        if config.id in self._configs:
            raise ValueError(f"Adapter 配置已存在: {config.id}")
        self._configs[config.id] = config
        return self._public_config(config)

    def update_config(
        self, adapter_id: str, *, name: str | None = None, enabled: bool | None = None
    ) -> dict[str, object]:
        """修改显示名称或启停状态。"""
        config = self._get_config(adapter_id)
        if name is not None:
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("Adapter name 不能为空")
            config.name = normalized_name
        if enabled is not None:
            config.enabled = enabled
        config.updated_at = now()
        return self._public_config(config)

    def delete_config(self, adapter_id: str) -> dict[str, object]:
        """删除逻辑配置及同 ID 的运行时 Adapter。"""
        config = self._get_config(adapter_id)
        if config.id in self._system_ids:
            raise ValueError(f"系统预配置 Adapter 不允许删除: {config.id}")
        public = self._public_config(config)
        del self._configs[config.id]
        self._adapters.pop(config.id, None)
        return public

    def list_configs(self) -> list[dict[str, object]]:
        """返回不含密钥的稳定排序配置列表。"""
        return [self._public_config(self._configs[key]) for key in sorted(self._configs)]

    def resolve(self, adapter_id: str) -> TargetAgentAdapter:
        """按 ID 获取 Adapter，未知 ID 返回可诊断错误。"""
        normalized_id = adapter_id.strip()
        config = self._configs.get(normalized_id)
        if config is not None and not config.enabled:
            raise AdapterDisabledError(f"Adapter 已停用: {normalized_id}")
        try:
            return self._adapters[normalized_id]
        except KeyError as error:
            raise AdapterNotFoundError(f"Adapter 未注册: {normalized_id}") from error

    def list_ids(self) -> tuple[str, ...]:
        """返回稳定排序的已注册 Adapter ID。"""
        return tuple(sorted(self._adapters))

    def _get_config(self, adapter_id: str) -> AdapterConfig:
        normalized_id = adapter_id.strip()
        try:
            return self._configs[normalized_id]
        except KeyError as error:
            raise AdapterNotFoundError(f"Adapter 配置不存在: {normalized_id}") from error

    def _public_config(self, config: AdapterConfig) -> dict[str, object]:
        return {
            "id": config.id,
            "name": config.name,
            "adapter_type": config.adapter_type.value,
            "enabled": config.enabled,
            "configured": config.id in self._adapters,
            "system": config.id in self._system_ids,
            "current_version_id": config.current_version_id,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat(),
        }


def create_default_registry() -> AdapterRegistry:
    """创建兼容当前 Demo 的默认 Adapter 注册表。"""
    return AdapterRegistry({"mock-rag": MockRagAdapter()})

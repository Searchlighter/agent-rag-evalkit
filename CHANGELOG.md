# Changelog

本项目记录对用户可见的能力、契约和兼容性变化。

## Unreleased

- 暂无。

## 0.5.0 - 2026-08-11

### Added

- LangGraph/Langfuse Trace、Badcase 工作台、回归比较和回答质量校验。
- MCP 风格评测、检索解释和 Badcase 汇总工具。
- Dify、RAGFlow 和通用 HTTP Adapter 契约示例。
- 合成 Mock RAG、双服务 Docker Compose、CI 和性能/成本基线。
- Issue Forms、Pull Request 模板、安全策略和完整社区协作说明。

### Changed

- README、架构、部署和 FAQ 按实际实现边界重写。
- 包与 FastAPI 元数据版本同步升级为 `0.5.0`。

### Compatibility

- 仍要求 Python 3.11+，CI 同时验证 Python 3.11 和 3.12。
- EvalKit 仍使用 `InMemoryRepository` 和同步执行，不提供数据迁移或持久化兼容保证。
- Adapter 统一输出结构保持兼容；Dify/RAGFlow 示例使用非流式响应。
- MCP 实现仍是 JSON-RPC/stdio 演示，并非官方 SDK 的完整协议兼容声明。

## 0.4.0 - 2026-08-11

- 增加包元数据、FastAPI 健康/就绪接口、初始 Docker 与 CI 骨架。

## 0.3.0

- 增加 MCP 风格工具发现与 Dify/RAGFlow 字段映射设计。

## 0.2.0

- 增加 Trace 标准化、Badcase 生命周期和回归集。

## 0.1.0

- 增加版本化评测集、确定性检索指标和 CSV 导出。

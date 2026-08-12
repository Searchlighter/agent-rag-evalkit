# 贡献指南

感谢参与 AgentRAG EvalKit。项目优先接受范围清晰、能够测试、不会夸大生产能力的改动。

## 开始之前

- Bug 和功能建议请使用 `.github/ISSUE_TEMPLATE` 中的表单。
- 安全漏洞不要提交公开 Issue，按 `SECURITY.md` 进行私密报告。
- 所有示例必须使用公开或人工合成数据，不得提交真实客户资料、个人信息、密钥和生产 Trace。
- 较大的数据模型、Adapter 契约或架构变更，建议先提交 Feature Request 讨论兼容边界。

## 本地开发

```bash
python -m pip install -e ".[dev]"
python -m compileall -q app scripts tests main.py
python -m ruff check app scripts tests main.py
python -m unittest discover -s tests -t . -v
python -m unittest tests.test_adapter_contracts -v
```

涉及容器时还应执行：

```bash
docker compose config --quiet
docker build --tag agent-rag-evalkit:local .
```

## 修改要求

### 核心逻辑

- 为新增行为添加 `unittest`，覆盖正常和错误输入。
- 保持 DatasetVersion、EvalRun、Badcase 等状态约束明确。
- 指标算法必须写明输入、空值和分母口径。

### Adapter 与外部契约

- 尽量保持现有字段向后兼容。
- 示例响应只能使用合成数据。
- 修改 Dify、RAGFlow 或 HTTP 映射时，必须更新 `tests/test_adapter_contracts.py` 和对应文档。
- 不在代码或测试中硬编码真实 Token。

### 文档与性能数据

- 对外可见的 API、Schema、CLI 或兼容性变化需要更新 `CHANGELOG.md`。
- 性能结果必须同时说明环境、样本量、轮次、并发和限制，不能把 Mock 数据写成生产容量。

## Pull Request

PR 应只解决一个明确问题，并填写 `.github/PULL_REQUEST_TEMPLATE.md`：

1. 关联 Issue 或解释独立修改原因。
2. 说明实现、测试和兼容性影响。
3. 确认不包含敏感数据。
4. 确保 CI 的质量、契约和 Docker Job 全部通过。

维护者可能要求拆分过大的 PR、补充失败用例，或对破坏性契约变更提供迁移方案。

## 行为规范与许可

参与项目即表示同意遵守 `CODE_OF_CONDUCT.md`。贡献内容将按照仓库 `LICENSE` 中的 MIT License 发布。

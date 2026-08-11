# 常见问题

## 这是一个可直接上线的企业级 RAG 平台吗？

不是。它是面向评测、诊断和回归的工程 Demo。当前数据保存在进程内，运行方式为同步执行，尚未实现生产所需的持久化、分布式任务、租户隔离和完整权限体系。

## 为什么 Demo 指标可能是满分？

默认 Mock Adapter 会返回与合成样本预期证据匹配的结果，用于证明数据导入、指标计算、Trace 和回归链路正确，而不是证明真实业务检索效果。

## 是否需要安装模型或配置 API Key？

运行 `python main.py` 和测试不需要。只有连接真实 Dify、RAGFlow、Langfuse 或自定义 RAG 服务时，才需要相应地址和凭据。

## 能评估哪些指标？

当前包含 Recall@K、MRR、引用覆盖率和空检索率，并支持关键词、正则、基础 JSON Schema、引用存在性/有效性等确定性检查。人工可对单条结果打 0～5 分并记录结论。

## 能否使用 LLM-as-a-Judge？

当前没有内置，以避免 Demo 在无 API Key 时失去可复现性。后续可通过质量检查扩展点增加 Judge，但应保存模型、Prompt 版本、温度、原始评分与人工抽检结果。

## 如何连接 LangGraph？

可通过 `app/langgraph_adapter.py` 的 Callback 采集节点开始、结束和异常事件，再规范化为 TraceEvent。详见 `docs/langgraph-callback-adapter.md`。

## Dify 和 RAGFlow 是真实 Adapter 还是字段示意？

已经实现 HTTP 请求、鉴权头、超时重试、响应校验和字段映射，并使用 Fake HTTP 响应做端到端契约测试；仓库没有附带可用的外部服务和密钥，因此不代表已经通过某个真实部署版本的联网联调。

## MCP 服务是否兼容所有 MCP Client？

不能这样承诺。当前是 MCP 风格的 JSON-RPC/stdio 演示服务，重点展示工具发现、调用、鉴权、校验、超时和审计。若用于正式 MCP 生态，应迁移到官方 SDK 并增加协议兼容测试。

## Trace 会不会泄露密钥？

标准化过程会对常见敏感字段进行脱敏，但自动脱敏不能覆盖所有业务文本。生产环境仍需在采集前进行字段分级、内容过滤、访问控制和保留期限管理。

## 如何定位一次检索失败？

先查看 CaseResult 指标，再使用 `explain_retrieval` 检查完整候选、原始/重排分数、过滤原因、预期证据是否出现以及相关 Trace。系统会给出确定性诊断，但根因结论仍可在 Badcase 工作台中由人工修订。

## 如何做版本回归？

将已确认的 Badcase 组成 Regression Suite，分别执行基线版本和候选版本，再生成 RegressionComparison。结果分为 passed、regressed、not_evaluable 和 needs_manual_review，避免将缺失结果误判为通过。

## 基准脚本的结果能写成生产性能吗？

不能。`scripts/benchmark.py` 只提供本机、内存、合成数据条件下的可复现开发基线。正式性能数据必须同时说明硬件、Python 版本、进程数、样本规模、目标服务、并发、轮次和统计口径。

## 数据为什么没有持久化？

这是 Demo 为降低启动成本作出的选择。生产化建议让 Repository 接口落地到 PostgreSQL，将大对象放到对象存储，并使用队列和 Worker 执行长任务。

## 可以把公司文档放进示例数据吗？

不建议。仓库只应保存合成或充分脱敏的数据，禁止提交 API Key、Cookie、个人信息、客户文档和生产 Trace。

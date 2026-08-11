# 示例数据说明

本目录中的内容全部为人工编写的合成数据，仅用于演示 AgentRAG EvalKit，不来源于任何真实公司制度、客户文档或用户记录。

| 文件 | 用途 |
| --- | --- |
| `enterprise_eval_cases.jsonl` | 4 条企业知识库评测样本，覆盖报销、采购、IT 和人事场景 |
| `mock_knowledge_base.json` | 与评测样本配套的 4 条模拟知识切片 |

证据标识统一使用 `<document_id>#<chunk_id>`。修改知识库中的标识时，需要同步修改评测集的 `expected_evidence`，否则 Recall@K、MRR 和 Citation Coverage 会按不匹配处理。

禁止将真实 API Key、个人信息、内部制度或生产 Trace 添加到本目录。

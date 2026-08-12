# 示例数据与 HTTP 演示

## 可视化 Dashboard

安装项目并启动 FastAPI：

```bash
pip install -e .
uvicorn app.main:app --reload
```

浏览器访问 <http://127.0.0.1:8000/dashboard>。应用启动时会执行一组包含 6 条脱敏合成 Case 的确定性评测，其中 4 条命中期望证据，2 条形成 `wrong_retrieval` Badcase。Dashboard 展示的 Recall@K、MRR、引用覆盖率、Case 结果和诊断信息均来自 `/api/v1/dashboard/snapshot`，不是写死在页面中的指标。

该演示用于展示操作和数据流，不代表真实业务准确率。真实系统验证需要通过 HTTP、Dify 或 RAGFlow Adapter 接入独立目标系统。

该演示使用 4 条人工合成企业制度和 4 条配套问题，证明 EvalKit 可以通过正式 HTTP Adapter 评测一个独立 RAG 服务。它不调用大模型，也不需要 API Key。

## 数据文件

- `sample_data/enterprise_eval_cases.jsonl`：评测问题、预期答案、预期证据、标签和规则校验条件。
- `sample_data/mock_knowledge_base.json`：模拟 RAG 使用的知识内容镜像，便于阅读和扩展。

所有文本均为人工合成，不对应真实企业制度。

## 运行

安装项目后，在两个终端中分别执行：

```bash
# Terminal 1：启动独立模拟 RAG 服务
uvicorn app.mock_rag_service:app --host 127.0.0.1 --port 8001
```

```bash
# Terminal 2：导入数据集并通过 HTTP Adapter 执行评测
python -m scripts.demo_http_eval
```

也可以指定目标地址或数据集：

```bash
python -m scripts.demo_http_eval \
  --endpoint http://127.0.0.1:8001/v1/query \
  --dataset sample_data/enterprise_eval_cases.jsonl
```

模拟服务的接口文档位于 <http://127.0.0.1:8001/docs>，健康检查位于 <http://127.0.0.1:8001/health>。

## 预期结果

脚本输出 JSON，包括运行状态、聚合指标以及每条 Case 的回答、证据、指标和规则校验结果。默认合成数据应该全部执行成功；这些结果仅说明 Demo 契约和指标链路正确。

## 扩展示例

1. 在 JSONL 中增加问题和 `expected_evidence`。
2. 在模拟知识库文件中增加对应文档，并同步更新 `app/mock_rag_service.py` 的合成切片。
3. 若要评测真实服务，保持 `/v1/query` 返回结构符合 `docs/http-adapter-contract.md`，再通过 `--endpoint` 指向它。

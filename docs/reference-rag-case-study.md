# Reference RAG 检索优化案例

## 目标

验证 EvalKit 能否通过正式 HTTP Adapter 评测一个实际构建索引的 RAG 服务，并利用失败样本对比检索策略。该案例不调用外部模型，便于在 GitHub Actions、Docker 和面试演示环境中离线复现。

## 数据范围

| 项目 | 数量 | 说明 |
| --- | ---: | --- |
| 知识切片 | 12 | 财务、采购、IT、人事和安全制度的人工合成内容 |
| 评测问题 | 24 | 每个主题包含直接表达或同义表达 |
| Top-K | 3 | 两种策略保持相同配置 |
| 数据级别 | synthetic-public | 不包含真实企业制度、用户数据或生产 Trace |

语料位于 `app/data/reference_corpus.json`，评测集位于 `sample_data/reference_eval_cases.jsonl`。

## 对比策略

### Baseline：BM25

对中英文文本生成词项，在内存中计算词频、文档频率、平均长度及 BM25 得分。该策略具备较强的字面匹配能力，但对完全不同的业务说法可能漏召回。

### Candidate：混合检索

在 BM25 基础上增加：

1. 显式、可审计的领域查询扩展，例如将“协议会签”扩展为“合同、审批、法务、财务”。
2. 采用 BLAKE2b 稳定哈希的字符 2～4 gram 稀疏向量。
3. 对 BM25 与向量余弦分数进行加权融合并统一排序。

哈希向量只用于无外部依赖的离线演示，不应描述为神经网络 Embedding。

## 实际结果

结果通过以下两个独立 HTTP 端点产生：

- `/v1/query/bm25`
- `/v1/query/hybrid`

EvalKit 使用 `HttpTargetAgentAdapter` 分别执行24条 Case：

| 指标 | BM25 | 混合检索 | 变化 |
| --- | ---: | ---: | ---: |
| Recall@K | 91.67% | 100.00% | +8.33个百分点 |
| MRR | 89.58% | 100.00% | +10.42个百分点 |
| Citation Coverage | 87.50% | 100.00% | +12.50个百分点 |
| Top-3 漏召回 | 2 | 0 | -2 |

机器可读结果保存在 `reports/reference-rag-comparison.json`。

## 发现的问题

BM25 在以下两条同义表达中未召回期望证据：

| Case | 查询 | 期望证据 | 诊断 |
| --- | --- | --- | --- |
| `ref_004` | 协议会签涉及哪些角色？ | `procurement-policy#contract` | 查询与文档缺少直接词项重合 |
| `ref_006` | 登录凭证失效了找谁处理？ | `it-handbook#password` | “登录凭证”与“账号密码”表达不一致 |

查询扩展补充领域词后，两条期望证据均排到混合检索首位。

## 复现方式

```bash
# Terminal 1
uvicorn app.reference_rag:app --host 127.0.0.1 --port 8002

# Terminal 2
python -m scripts.compare_reference_rag
```

第二条命令会覆盖生成 `reports/reference-rag-comparison.json`。报告移除了随机 Run ID，确保相同代码与数据重复执行时内容稳定。

## 适用边界

- 数据规模较小，不能证明大规模索引的性能和内存占用。
- 人工合成问题无法代表真实用户查询分布。
- 查询扩展词典需要按业务领域维护。
- 哈希向量可能发生碰撞，生产系统应评估真实 Embedding、向量数据库和 Reranker。
- 回答由首条证据抽取，不包含 LLM 生成质量和幻觉评估。

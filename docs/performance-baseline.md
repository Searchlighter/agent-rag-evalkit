# 性能与成本基线

## 1. 目的

该基准用于记录 AgentRAG EvalKit 在合成、内存场景下的开发基线，帮助发现明显性能回退。它不用于宣称生产吞吐、并发或模型成本。

## 2. 测试对象

- `EvalKitService` 同步评测主链路。
- `InMemoryRepository` 进程内存储。
- `MockRagAdapter` 确定性回答与检索结果。
- 数据集版本、指标计算、质量校验、Trace 标准化和结果保存。

不包含 HTTP、向量数据库、关系数据库、外部模型、Langfuse 网络导出和 Docker 开销。

## 3. 指标口径

| 指标 | 说明 |
| --- | --- |
| Run Latency | 单个 EvalRun 完成指定样本数的耗时 |
| P50/P95/P99 | 对多个 Run 耗时采用 nearest-rank 计算 |
| Wall Time | 所有测量 Run 在指定并发下的实际墙钟时间 |
| Cases/s | `总测量样本数 ÷ Wall Time`，仅代表本机合成链路 |
| Cost | 只记录实际可观察调用；Mock 无模型、Token 和 API 费用，均为 0 |

并发模式使用多个相互独立的 `EvalKitService` 和 `InMemoryRepository`，不代表同一生产实例的线程安全或共享存储吞吐。

## 4. 复现命令

```bash
python -m scripts.benchmark \
  --cases 100 \
  --rounds 20 \
  --concurrency 4 \
  --warmup 2 \
  --output benchmarks/baseline-p4-05.json
```

Windows PowerShell 可将命令写为一行。输出包含 Python、操作系统、CPU 数量、样本数、轮次、并发、延迟、吞吐和成本口径。

## 5. 基线记录

仓库中的 `benchmarks/baseline-p4-05.json` 是一次本地运行快照。判断回退时应在相同机器、Python 版本、参数和后台负载下重复执行至少 3 次，不应直接比较不同环境的绝对数值。

当前仓库基线快照：

| 项目 | 记录值 |
| --- | --- |
| 时间 | 2026-08-11 07:03:14 UTC |
| 环境 | Windows 10.0.19045、CPython 3.13.9、16 逻辑 CPU |
| 工作量 | 100 Case/Run × 20 Run，共 2000 Case |
| 并发/预热 | 4 / 2 Run |
| P50/P95/P99 | 5.360 / 38.500 / 49.391 ms 每 Run |
| Wall Time | 134.366 ms |
| 合成吞吐 | 14884.71 Case/s |
| 模型调用/Token/API成本 | 0 / 0 / ￥0.00 |

当前受限运行环境没有提供 CPU 型号，因此 JSON 中 `machine` 和 `processor` 明确记录为 `unknown`，没有猜测硬件信息。

建议将以下情况标记为待调查，而不是直接判定失败：

- 同环境 P95 连续多次上升超过团队约定阈值。
- 吞吐下降同时伴随 Case 失败或指标变化。
- 引入真实 Adapter 后成本字段缺失，而不是明确记录为未知。

## 6. 成本边界

当前基线的 API 成本为 0，是因为 `MockRagAdapter` 不调用模型。接入真实模型后必须从 Provider Usage 或账单获取 Token 与费用；未获得费用时应记录 `null/unknown`，不能继续沿用 0。

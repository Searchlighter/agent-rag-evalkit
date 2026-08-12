# Docker Compose 部署说明

## 1. 部署范围

项目提供一个面向本地复现的三服务 Compose：

| 服务 | 容器端口 | 主机端口 | 用途 |
| --- | ---: | ---: | --- |
| `evalkit-api` | 8000 | 8000 | EvalKit FastAPI、OpenAPI、健康检查 |
| `mock-rag` | 8001 | 8001 | 确定性合成 RAG 服务 |
| `reference-rag` | 8002 | 8002 | BM25 与混合检索 Reference RAG |

三个服务复用同一镜像，不包含数据库、Redis、队列或 LLM。EvalKit 数据仍位于进程内存，执行 `docker compose down` 或重启容器后会丢失。

## 2. 环境要求

- Docker Engine 24+ 或 Docker Desktop。
- Docker Compose v2（使用 `docker compose` 命令）。
- 最少 1.5 个可用 CPU、768 MB 可用内存和约 500 MB 磁盘空间。
- 主机端口 8000、8001、8002 未被占用。
- 首次构建可以访问基础镜像与 Python 包源。

Compose 为每个服务设置 `0.50 CPU / 256 MB` 上限，因此整体上限约为 `1.5 CPU / 768 MB`。该配置用于 Demo 资源保护，不代表压测结论或生产容量。

## 3. 一键启动

```bash
docker compose up --build -d
docker compose ps
```

等待三个服务均显示 `healthy` 后访问：

- EvalKit OpenAPI：<http://127.0.0.1:8000/docs>
- EvalKit Ready：<http://127.0.0.1:8000/ready>
- Mock RAG OpenAPI：<http://127.0.0.1:8001/docs>
- Mock RAG Health：<http://127.0.0.1:8001/health>
- Reference RAG OpenAPI：<http://127.0.0.1:8002/docs>
- Reference RAG Health：<http://127.0.0.1:8002/health>

随后可在宿主机执行完整 HTTP 评测：

```bash
python -m scripts.demo_http_eval
```

实时查看日志：

```bash
docker compose logs -f evalkit-api mock-rag
docker compose logs -f reference-rag
```

## 4. 停止与清理

```bash
docker compose down
```

该命令只停止并移除本项目容器和网络。当前 Compose 不创建数据卷，因此没有可恢复的持久化评测数据。

如需手动删除本地镜像，应先确认镜像名为 `agent-rag-evalkit-demo:local`，不要使用宽泛镜像清理命令。

## 5. 容器安全基线

- 镜像以 UID 10001 的 `evalkit` 非 root 用户运行。
- 根文件系统只读，仅提供 32 MB `/tmp` 临时文件系统。
- 丢弃 Linux Capabilities，并启用 `no-new-privileges`。
- `.dockerignore` 排除测试、文档、本地环境和潜在 `.env` 文件，减少构建上下文。
- 三个服务都有健康检查，EvalKit API 会等待 Mock RAG 与 Reference RAG 健康后启动。

这些设置是 Demo 基线，不替代生产环境的镜像扫描、签名、Secret 管理、网络策略和运行时监控。

## 6. 常见故障

### Docker daemon 未启动

若提示无法连接 `docker_engine` 或 Docker daemon，请先启动 Docker Desktop/Engine，再执行 Compose 命令。

### 端口被占用

修改 `docker-compose.yml` 端口映射左侧，例如将 `8000:8000` 改为 `18000:8000`；容器内部端口无需改变。

### 健康检查失败

```bash
docker compose ps
docker compose logs evalkit-api
docker compose logs mock-rag
```

重点检查依赖安装失败、端口冲突和宿主机代理/镜像源问题。

### 数据在重启后消失

这是当前 `InMemoryRepository` 的预期行为。需要持久化时，应先实现 PostgreSQL Repository 和迁移方案，而不是直接挂载 Python 进程目录。

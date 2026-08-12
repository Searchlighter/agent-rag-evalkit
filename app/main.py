"""FastAPI 应用工厂及存活、就绪探针。"""

from fastapi import FastAPI

from .api import create_router
from .dashboard import bootstrap_dashboard_demo, create_dashboard_router
from .repository import InMemoryRepository
from .service import EvalKitService


def create_app() -> FastAPI:
    """创建使用进程内仓储的可运行演示应用。"""
    app = FastAPI(title="AgentRAG EvalKit Demo", version="0.5.0")
    service = EvalKitService(InMemoryRepository())
    demo_run_id = bootstrap_dashboard_demo(service)
    app.include_router(create_router(service))
    app.include_router(create_dashboard_router(service, demo_run_id))

    @app.get("/health")
    def health() -> dict[str, str]:
        """报告进程存活状态。"""
        return {"status": "ok", "version": app.version}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """报告服务就绪状态及当前存储实现。"""
        return {"status": "ready", "storage": "in_memory"}

    return app


app = create_app()

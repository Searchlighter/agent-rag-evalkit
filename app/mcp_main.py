"""启动使用进程内仓储的 stdio 工具服务。"""

from .mcp_server import EvalKitMcpServer, serve_stdio
from .repository import InMemoryRepository
from .service import EvalKitService


if __name__ == "__main__":
    serve_stdio(EvalKitMcpServer(EvalKitService(InMemoryRepository())))

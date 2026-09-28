"""集成测试使用的临时本地 Uvicorn 服务。"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from threading import Thread
from time import monotonic, sleep
from typing import Iterator

import uvicorn
from fastapi import FastAPI


@contextmanager
def running_server(app: FastAPI) -> Iterator[str]:
    """在随机本地端口启动 Uvicorn，并在测试结束时可靠关闭。"""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="error",
            lifespan="off",
            access_log=False,
            ws="none",
        )
    )
    thread = Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = monotonic() + 5
    while not server.started and thread.is_alive() and monotonic() < deadline:
        sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        raise RuntimeError("local integration server failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("local integration server failed to stop")

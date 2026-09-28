"""EvalKit 主 API 到本地 Mock RAG 的真实 HTTP 集成测试。"""

from __future__ import annotations

import json
import socket
import unittest
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any, Iterator
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI

from app.adapter_registry import AdapterRegistry
from app.http_adapter import HttpTargetAgentAdapter
from app.ingestion import load_jsonl_cases
from app.main import create_app
from app.mock_rag_service import app as mock_rag_app


ROOT = Path(__file__).resolve().parents[1]


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


def request_json(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    """使用标准库发起真实 JSON HTTP 请求。"""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


class HttpApiIntegrationTests(unittest.TestCase):
    def test_main_api_executes_four_cases_through_real_http_mock_rag(self) -> None:
        cases = load_jsonl_cases(ROOT / "sample_data" / "enterprise_eval_cases.jsonl")
        with running_server(mock_rag_app) as mock_url:
            registry = AdapterRegistry({
                "local-mock-rag": HttpTargetAgentAdapter(
                    f"{mock_url}/v1/query", retries=0
                )
            })
            with running_server(create_app(registry)) as api_url:
                probe = request_json(
                    f"{api_url}/api/v1/adapters/test",
                    "POST",
                    {"adapter_id": "local-mock-rag", "question": cases[0]["question"]},
                )
                dataset = request_json(
                    f"{api_url}/api/v1/datasets",
                    "POST",
                    {"name": "real-http-integration", "owner_id": "integration-test"},
                )
                version = request_json(
                    f"{api_url}/api/v1/datasets/{dataset['id']}/versions",
                    "POST",
                    {"cases": cases, "actor_id": "integration-test"},
                )
                config = request_json(
                    f"{api_url}/api/v1/configs",
                    "POST",
                    {"name": "real-http-top2", "retrieval_k": 2},
                )
                run = request_json(
                    f"{api_url}/api/v1/eval-runs",
                    "POST",
                    {
                        "dataset_version_id": version["id"],
                        "config_id": config["id"],
                        "adapter_id": "local-mock-rag",
                    },
                )
                summary = request_json(
                    f"{api_url}/api/v1/eval-runs/{run['id']}/execute", "POST"
                )
                results = request_json(
                    f"{api_url}/api/v1/eval-runs/{run['id']}/results"
                )

        self.assertEqual("ok", probe["status"])
        self.assertTrue(probe["response_request_id_matches"])
        self.assertEqual("succeeded", summary["status"])
        self.assertEqual(4, summary["completed_case_count"])
        self.assertEqual(1.0, summary["metrics"]["recall_at_k"])
        self.assertEqual(4, len(results))
        self.assertTrue(all(item["status"] == "succeeded" for item in results))
        self.assertTrue(all(item["retrieval_ids"] for item in results))


if __name__ == "__main__":
    unittest.main()

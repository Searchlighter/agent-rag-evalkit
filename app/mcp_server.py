"""依赖较少的 JSON-RPC/stdio 工具服务，提供鉴权、校验、超时和审计。"""

from __future__ import annotations

import hmac
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from .adapter_contract import MockRagAdapter
from .service import EvalKitService
from .trace import redact_text


@dataclass(slots=True)
class McpSecurityConfig:
    """工具服务的可选共享 Token、执行超时和工具白名单。"""
    api_token: str | None = None
    timeout_seconds: float = 10.0
    allowed_tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class EvalKitMcpServer:
    """将评测、检索解释和 Badcase 汇总暴露为结构化工具。"""

    def __init__(
        self,
        service: EvalKitService,
        security: McpSecurityConfig | None = None,
    ) -> None:
        self.service = service
        self.security = security or McpSecurityConfig()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="evalkit-mcp")

    def list_tools(self) -> list[dict[str, Any]]:
        """返回经过白名单过滤的工具定义和输入 Schema。"""
        tools = [
            {
                "name": "evaluate_rag",
                "description": "Create, execute, or query an evaluation run.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["create", "create_and_execute", "execute", "get"],
                        },
                        "eval_run_id": {"type": "string"},
                        "dataset_version_id": {"type": "string"},
                        "config_id": {"type": "string"},
                        "adapter_id": {"type": "string"},
                        "budget_limit": {"type": "number"},
                    },
                    "required": ["operation"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "explain_retrieval",
                "description": "Return retrieval trace and deterministic diagnostic suggestions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "eval_run_id": {"type": "string"},
                        "case_id": {"type": "string"},
                    },
                    "required": ["eval_run_id", "case_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_badcase_summary",
                "description": "Aggregate Badcases by version, tags, severity, category, and status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "eval_run_id": {"type": "string"},
                        "severity": {"type": "string"},
                        "dataset_version_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "tag_match": {
                            "type": "string",
                            "enum": ["any", "all"],
                        },
                        "category": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        ]
        if self.security.allowed_tools is None:
            return tools
        return [item for item in tools if item["name"] in self.security.allowed_tools]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        auth_token: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """鉴权并校验一次工具调用，在超时后返回可审计错误。"""
        request_id = request_id or f"mcp-{uuid.uuid4().hex[:12]}"
        auth_error = self._authorize(auth_token, request_id)
        if auth_error:
            return auth_error
        tool = next((item for item in self.list_tools() if item["name"] == name), None)
        if tool is None:
            return self._audited_error(
                "tool_not_found", f"Unknown or forbidden tool: {name}", request_id, name
            )
        validation_error = _validate_arguments(arguments, tool["inputSchema"])
        if validation_error:
            return self._audited_error(
                "invalid_arguments", validation_error, request_id, name
            )

        future = self._executor.submit(self._execute_tool, name, arguments)
        try:
            result = future.result(timeout=self.security.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return self._audited_error(
                "tool_timeout",
                f"Tool exceeded {self.security.timeout_seconds} seconds",
                request_id,
                name,
                retryable=True,
            )
        except (KeyError, ValueError) as error:
            return self._audited_error(
                "invalid_request", redact_text(error), request_id, name
            )
        except Exception as error:
            return self._audited_error(
                "internal_error", redact_text(error), request_id, name, retryable=True
            )

        self.service.record_external_audit(
            "mcp_tool_succeeded",
            "mcp_tool",
            name,
            "mcp-client",
            {"request_id": request_id},
        )
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, default=str, ensure_ascii=False),
            }],
            "_meta": {"request_id": request_id},
        }

    def dispatch(self, request: object, *, auth_token: str | None = None) -> dict[str, Any]:
        """处理一条 JSON-RPC 风格请求，并始终返回可序列化响应。"""
        if not isinstance(request, dict):
            request_id = f"mcp-{uuid.uuid4().hex[:12]}"
            result = self._audited_error(
                "invalid_request", "request must be an object", request_id, "unknown"
            )
            return {"jsonrpc": "2.0", "id": None, "result": result}
        request_id = str(request.get("id") or f"mcp-{uuid.uuid4().hex[:12]}")
        token = auth_token or request.get("auth_token")
        auth_error = self._authorize(token, request_id)
        if auth_error:
            result: Any = auth_error
        elif request.get("method") == "tools/list":
            result = {"tools": self.list_tools()}
        elif request.get("method") == "tools/call":
            params = request.get("params", {})
            if not isinstance(params, dict):
                result = self._audited_error(
                    "invalid_request", "params must be an object", request_id, "unknown"
                )
            else:
                result = self.call_tool(
                    params.get("name", ""),
                    params.get("arguments", {}),
                    auth_token=token,
                    request_id=request_id,
                )
        else:
            result = self._audited_error(
                "method_not_found",
                f"Unknown method: {request.get('method')}",
                request_id,
                "unknown",
            )
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "evaluate_rag":
            return self._evaluate_rag(arguments)
        if name == "explain_retrieval":
            return self.service.explain_retrieval(
                arguments["eval_run_id"], arguments["case_id"]
            )
        if name == "get_badcase_summary":
            return self.service.get_badcase_summary(
                eval_run_id=arguments.get("eval_run_id"),
                severity=arguments.get("severity"),
                dataset_version_id=arguments.get("dataset_version_id"),
                tags=arguments.get("tags"),
                tag_match=arguments.get("tag_match", "any"),
                category=arguments.get("category"),
                status=arguments.get("status"),
            )
        raise KeyError(name)

    def _evaluate_rag(self, arguments: dict[str, Any]) -> dict[str, Any]:
        operation = arguments["operation"]
        if operation in {"create", "create_and_execute"}:
            dataset_version_id = _required_string(arguments, "dataset_version_id")
            config_id = _required_string(arguments, "config_id")
            run = self.service.create_eval_run(
                dataset_version_id,
                config_id,
                str(arguments.get("adapter_id") or "mock-rag"),
                arguments.get("budget_limit"),
            )
            summary = (
                self.service.execute_eval_run(run.id, MockRagAdapter())
                if operation == "create_and_execute"
                else self.service.get_run_summary(run.id)
            )
        elif operation == "execute":
            run_id = _required_string(arguments, "eval_run_id")
            summary = self.service.execute_eval_run(run_id, MockRagAdapter())
        elif operation == "get":
            run_id = _required_string(arguments, "eval_run_id")
            summary = self.service.get_run_summary(run_id)
        else:
            raise ValueError(f"Unsupported evaluate_rag operation: {operation}")
        return {
            "operation": operation,
            "run_id": summary["run_id"],
            "status": summary["status"],
            "summary": summary,
        }

    def _authorize(
        self, auth_token: object, request_id: str
    ) -> dict[str, Any] | None:
        expected = self.security.api_token
        if expected is None:
            return None
        if not isinstance(auth_token, str) or not hmac.compare_digest(auth_token, expected):
            return self._audited_error(
                "unauthorized", "Invalid or missing API token", request_id, "auth"
            )
        return None

    def _audited_error(
        self,
        code: str,
        message: str,
        request_id: str,
        tool_name: str,
        *,
        retryable: bool = False,
    ) -> dict[str, Any]:
        self.service.record_external_audit(
            "mcp_tool_failed",
            "mcp_tool",
            tool_name,
            "mcp-client",
            {"request_id": request_id, "code": code, "retryable": retryable},
        )
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                    "request_id": request_id,
                }),
            }],
            "_meta": {"request_id": request_id},
        }

    def close(self) -> None:
        """停止接收新任务，并取消尚未开始的后台调用。"""
        self._executor.shutdown(wait=False, cancel_futures=True)


def _validate_arguments(arguments: object, schema: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    properties = schema.get("properties", {})
    for field_name in schema.get("required", []):
        if field_name not in arguments or arguments[field_name] in ("", None):
            return f"{field_name} is required"
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"unknown argument(s): {', '.join(unknown)}"
    for field_name, value in arguments.items():
        field_schema = properties.get(field_name, {})
        expected = field_schema.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{field_name} must be a string"
        if expected == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return f"{field_name} must be a number"
        if expected == "array":
            if not isinstance(value, list):
                return f"{field_name} must be an array"
            item_type = field_schema.get("items", {}).get("type")
            if item_type == "string" and not all(isinstance(item, str) for item in value):
                return f"{field_name} items must be strings"
        if field_schema.get("enum") and value not in field_schema["enum"]:
            return f"{field_name} must be one of: {', '.join(field_schema['enum'])}"
    return None


def _required_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required for this operation")
    return value.strip()


def serve_stdio(server: EvalKitMcpServer) -> None:
    """逐行处理标准输入，并在 EOF 时释放执行线程池。"""
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                print(json.dumps(server.dispatch(request), ensure_ascii=False), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "parse error"},
                }), flush=True)
    finally:
        server.close()

# MCP Infrastructure

EvalKit provides a dependency-light JSON-RPC/stdio MCP-style infrastructure
layer for the Demo.

It includes:

- optional API-token authentication using constant-time comparison;
- per-server tool allowlists;
- JSON-Schema-style required-field, type, and unknown-field validation;
- bounded tool execution with structured timeout errors;
- structured error codes, retryable flags, and request IDs;
- success/failure audit events.

Authentication is disabled by default to preserve the local Demo. Configure
McpSecurityConfig with an API token before exposing the process outside a
trusted development environment.

The timeout uses a thread future. It returns control when the deadline is
exceeded but cannot forcibly terminate arbitrary Python code already running
inside the worker. Production execution should use isolated worker processes
or a job queue with cancellation support.

This module mirrors MCP tool discovery and tool-call semantics but does not
claim full official SDK or transport compliance.

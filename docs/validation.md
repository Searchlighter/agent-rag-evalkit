# Validation Record

## Local checks completed

- Python compile check for app, scripts, and Demo entry point.
- Unit test suite covering evaluation, CSV import/export, Trace, Badcase,
  regression, MCP-style tool discovery, readiness metadata, and benchmark.
- Synthetic end-to-end Demo execution.
- Dockerfile/Compose static contract tests.
- Docker Compose configuration parsing via `docker compose config --quiet`.
- CI workflow static contract tests and local Adapter contract tests.
- Ruff 0.16.2 lint using the same command and explicit rule set as CI.
- Synthetic baseline with environment, 2,000 measured cases,
  concurrency 4, latency percentiles, throughput, and explicit zero-cost scope.
- Community health contract tests for license, version metadata, Issue
  Forms, PR template, contribution guide, conduct policy, and security policy.

## Not executed locally

- Image build and container startup in the current session because the
  Docker CLI is installed but the Docker daemon is not running. A daemon-backed
  build remains an environment-dependent release check.
- GitHub Actions execution, because it requires a pushed repository.
- Real external integration tests for LangGraph, Dify, RAGFlow, Langfuse, and
  an official MCP SDK.

The synthetic benchmark is intentionally excluded from release claims because
it measures only the in-memory Mock Adapter path.

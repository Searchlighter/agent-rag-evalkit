# MCP Tool: evaluate_rag

The evaluation tool supports four operations:

- create: create a queued EvalRun;
- create_and_execute: create and synchronously execute a run with the Demo Mock Adapter;
- execute: execute an existing queued run;
- get: query an existing run summary.

Create operations require dataset_version_id and config_id. Execute/get require
eval_run_id. adapter_id defaults to mock-rag, and budget_limit is optional.

Every successful call returns operation, run_id, status, and the complete run
summary. Invalid operation-specific arguments return a structured
invalid_request error and are included in the MCP audit trail.

The local Demo uses MockRagAdapter. A production tool should resolve adapter_id
through an authenticated adapter registry instead of accepting arbitrary
network destinations from tool arguments.

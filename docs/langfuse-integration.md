# Langfuse Integration

Install the optional dependency:

    pip install -e ".[observability]"

Create the Langfuse client from runtime environment variables, then pass it to
LangfuseTraceExporter. The exporter creates one root span per evaluation case
and nested retriever, generation, tool, or generic span observations.

For generation observations, model, usage_details, and cost_details can be
attached. EvalRun ID, Case ID, CaseResult ID, normalized event ID, source, and
duration are included as metadata.

Custom trace IDs are converted to a deterministic 32-character hexadecimal
identifier for W3C trace-context compatibility. Export uses already-normalized
and redacted TraceEvent data.

The base installation does not require Langfuse and does not send any network
requests.

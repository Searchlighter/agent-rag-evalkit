# MCP Tool: get_badcase_summary

The Badcase summary tool supports these filters:

- EvalRun ID;
- dataset-version ID;
- one or more Case tags with any/all matching;
- severity;
- category;
- lifecycle status.

The response includes totals grouped by category, status, severity,
dataset-version ID, and tag. Item details include the source EvalRun,
dataset-version ID, Case ID, tags, category, severity, and status.

Tag filters are resolved from the immutable DatasetVersion used by the source
EvalRun, so later dataset versions do not silently change historical Badcase
summaries.

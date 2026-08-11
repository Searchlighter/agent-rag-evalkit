# MCP Tool: explain_retrieval

The retrieval explanation tool analyzes one evaluated case using its stored candidates and
normalized Trace.

The response includes:

- candidate evidence IDs, document/chunk IDs, original scores, and ranks;
- optional rerank scores from candidate metadata;
- selected/filtered flags and filter reasons;
- whether each candidate matches expected evidence;
- missing expected evidence and unexpected retrievals;
- retrieval/rerank Trace events;
- deterministic diagnosis and troubleshooting suggestion.

The target Adapter can provide rerank_score, filtered, and filter_reason in a
RetrievedChunk metadata object. If those values are absent, EvalKit reports
them as unknown rather than inventing them.

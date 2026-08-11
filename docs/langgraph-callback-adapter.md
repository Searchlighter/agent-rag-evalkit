# LangGraph Callback Adapter

Use LangGraphCallbackAdapter when the target graph exposes an invoke-style
function. The adapter passes a callbacks list and a configurable thread_id:

    adapter = LangGraphCallbackAdapter(graph.invoke)
    result = adapter.invoke("question", request_id="run-id:case-id")

The graph state must include answer. It may include citations, retrievals, and
events. Each retrieval needs document_id and chunk_id; score and rank are
optional.

The callback collector records retriever, LLM, tool, and error events. The TraceEvent
normalizes these into the persisted TraceEvent schema.

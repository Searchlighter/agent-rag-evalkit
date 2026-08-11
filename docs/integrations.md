# Integration Examples

## MCP-style stdio demo

The demo exposes three tools: evaluate_rag, explain_retrieval, and
get_badcase_summary.

Run:

    python -m app.mcp_main

Then send one JSON-RPC line:

    {"jsonrpc":"2.0","id":1,"method":"tools/list"}

This implementation intentionally avoids an MCP SDK. A production server should
use the official SDK and add authentication, authorization, request timeout,
and persistent storage.

## Dify HTTP workflow example

Use an HTTP Request node after a Dify workflow finishes. Map its answer,
citations, retrieval IDs, and callback events to the EvalKit adapter schema:

    {
      "request_id": "{{sys.conversation_id}}",
      "answer": "{{workflow.answer}}",
      "citations": "{{workflow.citations}}",
      "retrievals": "{{workflow.retrievals}}",
      "events": "{{workflow.trace_events}}"
    }

The adapter contract requires answer, citations, retrievals, and a stable
request_id. The Trace normalizer maps retrieval, rerank, llm, tool, and error
events into diagnostic Trace records.

## RAGFlow field mapping

| EvalKit field | RAGFlow mapping |
| --- | --- |
| question | user query |
| answer | generated answer |
| retrievals.document_id | document ID |
| retrievals.chunk_id | chunk ID |
| retrievals.score | retrieval or rerank score |
| citations | document_id#chunk_id list |

For reproducibility, store the RAGFlow knowledge-base version and model name in
the Adapter event metadata.

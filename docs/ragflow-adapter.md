# RAGFlow Adapter

The RAGFlow integration uses its OpenAI-compatible non-streaming chat endpoint:

    POST /api/v1/openai/{chat_id}/chat/completions

The request enables extra_body.reference and reference metadata. The adapter
maps message.reference.chunks into EvalKit retrievals:

- document_id becomes the evidence document ID;
- chunk id becomes the EvalKit chunk ID;
- similarity becomes the retrieval score;
- vector/term similarity, dataset, document name/content, and document metadata
  remain available as candidate metadata;
- citations use document_id#chunk_id.

The parser accepts reference.chunks as either an object or list to make the
example tolerant of exported result variants. examples/ragflow_chat_completion.json
contains synthetic data only.

The API key is a runtime secret and must not be committed. Tests use a fake
HTTP response and do not contact a RAGFlow deployment.

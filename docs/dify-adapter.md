# Dify Chat Adapter

The Dify integration provides a blocking Chat App adapter using POST /chat-messages.
It sends inputs, query, response_mode=blocking, user, and an optional
conversation_id with Bearer API-key authentication.

The adapter maps metadata.retriever_resources into EvalKit retrievals:

- document_id becomes the evidence document ID;
- segment_id becomes the chunk ID;
- score and position become retrieval score and rank;
- remaining fields are preserved as metadata;
- citations use document_id#segment_id.

Dify message/conversation IDs and usage metadata are included in normalized
events. examples/dify_blocking_response.json contains synthetic data only.

Do not place a Dify API key in source control. Instantiate DifyChatAdapter from
runtime secrets. This Demo tests parsing and request construction with a fake
HTTP response; it does not contact a Dify server.

If the response omits `answer` or `metadata.retriever_resources`, the adapter
preserves those omissions as `missing_fields`. EvalKit records the case as
`not_evaluable` instead of treating missing evidence as a valid empty retrieval.

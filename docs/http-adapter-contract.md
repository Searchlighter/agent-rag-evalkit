# HTTP Agent Adapter Contract

POST a JSON request to the configured endpoint:

    {"question":"employee expense policy","request_id":"run_x:case_y"}

The service must return JSON with this shape:

    {
      "request_id": "run_x:case_y",
      "answer": "answer text",
      "citations": ["document-id#chunk-id"],
      "retrievals": [
        {"document_id":"document-id","chunk_id":"chunk-id","score":0.91,"rank":1}
      ],
      "events": [{"type":"retrieval","latency_ms":18}]
    }

The adapter uses POST, an X-Request-ID header, optional Bearer authorization,
a configurable timeout, and immediate retries for network/HTTP failures.
Never put a bearer token in source code; provide it at runtime.

Configuration and response validation rules:

- `endpoint` must be an absolute HTTP/HTTPS URL without embedded credentials or a fragment.
- `timeout_seconds` must be greater than 0 and no more than 300; retries are limited to 0-5.
- Bearer tokens and request IDs must not contain control characters used for header injection.
- A returned `request_id` must match the request; omitted request IDs inherit the request value.
- Citations must be a list of non-empty strings.
- Each retrieval requires string `document_id` and `chunk_id`, a finite numeric score,
  a positive integer rank, and object metadata.

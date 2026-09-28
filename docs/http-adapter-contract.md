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
a configurable timeout, exponential-backoff retries, and a per-adapter circuit breaker.
Never put a bearer token in source code; provide it through an environment variable:

    $env:EVALKIT_HTTP_BEARER_TOKEN = "replace-me"  # PowerShell

    adapter = HttpTargetAgentAdapter(
        endpoint="https://agent.example.com/query",
        bearer_token_env="EVALKIT_HTTP_BEARER_TOKEN",
    )

The environment variable is resolved when the adapter is created. Its value is excluded
from the adapter representation and is never included in connection-test API responses.

Configuration and response validation rules:

- `endpoint` must be an absolute HTTP/HTTPS URL without embedded credentials or a fragment.
- `timeout_seconds` must be greater than 0 and no more than 300; retries are limited to 0-5.
- Retries apply to timeouts, network failures, HTTP 408/425/429, and HTTP 5xx responses.
  Other HTTP 4xx responses fail immediately. Backoff doubles after each failed attempt.
- Consecutive failed calls open the circuit. Calls fail fast until the recovery window elapses;
  the first successful recovery call closes the circuit and resets the failure count.
- Transport errors are classified as timeout, network, HTTP status, or circuit-open errors.
- Failed cases expose a `failed_case_count`; stored result, Trace, and CSV error text is
  credential-redacted and length-limited before it reaches an API response or export.
- Bearer tokens and request IDs must not contain control characters used for header injection.
- Configure either `bearer_token_env` or the legacy direct token argument, never both.
- A returned `request_id` must match the request; omitted request IDs inherit the request value.
- Citations must be a list of non-empty strings.
- Each retrieval requires string `document_id` and `chunk_id`, a finite numeric score,
  a positive integer rank, and object metadata.
- A field that is omitted is recorded as missing and produces a `not_evaluable` case;
  an explicitly returned empty list remains an evaluable empty result.

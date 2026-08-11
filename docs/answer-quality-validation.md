# Answer Quality Validation

EvalKit provides deterministic checks configured through EvalCase metadata:

- required_keywords: every keyword must appear, case-insensitively;
- required_patterns: every regular expression must match;
- require_citations: at least one citation must be returned;
- citation validity: each citation must exist in the retrieved evidence IDs;
- answer_json_schema: validates a dependency-free subset containing type,
  required, properties, and primitive property types.

Case results can also receive a human score from 0 to 5, reviewer, note, and
review timestamp. The review action is included in the audit log.

The JSON Schema implementation is intentionally limited. Production use should
replace it with a standards-compliant validator when advanced schema keywords
are required.

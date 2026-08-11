# Badcase Regression Suite

The regression workflow converts selected Badcases into a versioned comparison scope. A suite
keeps source Badcase IDs, case IDs, and their original EvalRun IDs.

Each baseline/candidate comparison stores:

- comparison ID and suite ID;
- baseline and candidate EvalRun IDs;
- baseline and candidate dataset-version IDs;
- per-case baseline/candidate deterministic scores;
- outcome and diagnostic reason.

Supported outcomes:

- passed: candidate score is equal to or better than baseline;
- regressed: candidate score decreased;
- not_evaluable: a result is missing or candidate execution failed;
- needs_manual_review: deterministic metrics are absent or answer checks failed.

Comparison records can be listed by suite and queried later. Production use
should persist them in PostgreSQL and enforce access control.

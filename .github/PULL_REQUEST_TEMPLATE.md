## Summary

Describe the problem and the smallest change that solves it.

## Related issue

Closes #

## Change type

- [ ] Core evaluation or data model
- [ ] Adapter or external contract
- [ ] API, MCP-style tool, or deployment
- [ ] Documentation or community configuration

## Validation

List the exact commands run and relevant results.

```text
python -m ruff check app scripts tests main.py
python -m unittest discover -s tests -v
```

## Compatibility and safety

- [ ] I documented externally visible API, schema, or Adapter changes.
- [ ] I added or updated tests for changed behavior.
- [ ] Examples contain only synthetic or public data.
- [ ] The change contains no API keys, credentials, personal data, customer documents, or production traces.
- [ ] I described migration or backward-compatibility impact, or confirmed there is none.

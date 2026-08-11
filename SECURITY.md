# Security Policy

## Supported version

This portfolio Demo currently supports the latest repository revision only. It is not a production security product and does not provide security fixes for historical Demo versions.

## Reporting a vulnerability

Do not open a public Issue for vulnerabilities, leaked credentials, or exposed data. After the repository is published, use GitHub Private Vulnerability Reporting from the repository Security tab when enabled; otherwise contact the repository owner through a private channel listed on their GitHub profile.

Include the affected revision, impact, minimal synthetic reproduction, and suggested mitigation. Do not include real customer data, personal information, API keys, or production traces.

Maintainers should acknowledge a report when practical, validate its scope, prepare a fix without public disclosure, and publish a concise advisory after remediation. No response-time SLA is promised for this portfolio Demo.

## Security boundaries

The current in-memory Demo lacks production identity management, tenant isolation, durable audit storage, secret management, and distributed deployment controls. See `README.md` and `docs/architecture.md` before exposing it to untrusted networks.

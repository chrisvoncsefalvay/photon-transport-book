# Security policy

Please report suspected credential exposure, release-boundary failures, dependency vulnerabilities, or unsafe handling of medical data privately to `chris@chrisvoncsefalvay.com`. Do not open a public issue containing a secret, private repository path, patient data, or exploitable detail.

The public exporter treats the manifest as the authority, rejects forbidden paths, and scans for common secret and local-path patterns. This is a practical guardrail, not a substitute for human review. Never commit credentials or identifiable medical data, even if a later scan should exclude them.

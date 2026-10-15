# AutoResearch backend security model

Provider credentials remain exclusively in the backend process environment.
TOML stores environment-variable names, rejects inline secrets, and provider
status responses expose readiness, provider kind, model role, and redacted base
URLs—not keys. Backend errors, events, durable messages, and artifacts pass
through the redaction boundary for authorization headers, credential-bearing
URLs, token-like query values, and configured secret values.

Generic agent tools are contained to an explicitly allowed repository root.
Absolute paths, parent traversal, symlink escapes, control characters, shell
commands, and non-allowlisted argv vectors are rejected. Typed experiment
actions are resolved by policy and executed without a shell; exact workspace,
specification checksum, and approval checks remain separate from generic file
or terminal permissions.

All mutations use durable idempotency where applicable. SSE cursors are parsed
as non-negative integers, event history is replayed from durable IDs, and the
client treats malformed or stale events as warnings rather than evidence.
Cancellation is checked again before a pending approved tool executes, so a
late permission decision cannot write after session cancellation.

The terminal client never reads provider-key environment variables, browser
storage, or raw request headers. It may read one configured backend-auth token
environment variable; provider credential names are rejected by client config.
There is no browser frontend in this release, so browser-storage and DOM-XSS
controls are not applicable to the product surface.

Security QA commands:

```bash
cd /path/to/autoresearch
rg -n 'shell\s*=\s*True|os\.system|eval\(|exec\(|Popen' src tests
.venv/bin/python -m pytest -q tests/qa/test_security_boundaries.py tests/test_provider_contract.py

cd <companion-terminal-client-dir>
rg -n 'localStorage|sessionStorage|dangerouslySetInnerHTML|innerHTML|API_KEY' packages/autoresearch-cli/src packages/autoresearch-cli/test
```

Expected subprocess matches are shell-less `create_subprocess_exec` calls in
typed policy-controlled paths and the allowlisted generic terminal tool. Test
fixtures may contain deliberately fake secrets to prove redaction; they are
not production credentials.

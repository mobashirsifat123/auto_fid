# AutoResearch v0.1.0-alpha release platform

## One command

After installation, run:

```bash
autoresearch
```

The launcher reads `~/.config/autoresearch/config.toml` on Linux and the
OS-equivalent application-support directory on macOS/Windows. It starts a
loopback-only Python daemon, waits for `/v1/health`, checks API major-version
compatibility, and then launches the packaged `autoresearch-tui` binary. Runtime
does not use Bun, `npx`, or a provider key.

Useful commands:

```bash
autoresearch version
autoresearch setup --allowed-root /path/to/repository
autoresearch doctor --json
autoresearch status --json
autoresearch run --prompt "Summarize the current research workspace" --jsonl
autoresearch diagnostics --output /tmp/autoresearch-diagnostics.zip
autoresearch update --source /path/to/release-directory
autoresearch rollback
autoresearch uninstall
```

`setup` and `login` store only environment-variable names and secret-free
provider configuration references. They never accept raw provider secrets.

## Build and install

From a source checkout with Python 3.11+ and Bun available at build time:

```bash
.venv/bin/python scripts/build_release.py
/bin/bash dist/release/autoresearch-0.1.0-alpha.1-<host>/install.sh \
  dist/release/autoresearch-0.1.0-alpha.1-<host>
```

For backend-only packaging CI, use `--skip-tui`. The host builder supports an
existing compiled binary with `--tui-binary`. Release files are hashed in
`manifest.json`; the current local artifact is intentionally unsigned because
no signing identity was supplied. A production distribution must verify the
manifest and add a platform signing step before publication.

## Update and rollback

`autoresearch update --source RELEASE_DIR` verifies every manifest checksum and
copies the verified tree into a unique state staging directory. Add
`--activate` to install the host TUI after staging; the previous TUI is retained
under product state. Run `autoresearch doctor` and the versioned health check
after activation. `autoresearch rollback` restores the newest retained TUI;
backend wheels remain managed by the Python environment and are not silently
replaced by a client-only update.

## CI

Backend CI runs tests, compilation, dependency checks, the offline ResearchBench
gate, and backend packaging. CLI CI runs locked installation, typecheck, the
deterministic terminal suite, and production bundling. OpenAPI freshness is
checked by `packages/autoresearch-cli/scripts/check-contract.ts` whenever a
backend checkout is supplied; local combined validation runs it unconditionally.

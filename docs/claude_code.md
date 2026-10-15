# Claude Code research modes

Two integration paths exist:

1. **Agent skills (recommended):** install the skill bridges with
   `scripts/install_skills.sh` (Claude Code) or
   `autoresearch skill install --target all` (Claude Code and OpenCode).
   The skills invoke the stable `autoresearch skill …` CLI and interpret its
   JSON output — see `src/autoresearch/skills/` for the canonical definitions.
2. **Agent-mode bridge:** the durable conversational bridge below, which any
   agent host can drive over the local API.

The bridge provides three research modes:

* `/review <request>` starts Deep Literature Review, including evidence verification, ranking, and cited research-gap analysis.
* `/novelty <request>` starts evidence-backed novelty candidate generation for a validated research gap.
* `/experiment <request>` starts experiment design for an existing novelty candidate.

Each mode uses `autoresearch agent`, which creates or reuses a durable session per repository and mode, starts the backend turn without waiting for completion, and prints the session and turn identifiers. Session state is stored under AutoResearch's local state directory, not in the repository and never contains provider credentials.

With the package installed:

```bash
cd /path/to/autoresearch
autoresearch agent --mode review --prompt "<request>" --wait
```

Use one mode at a time. A mode remains part of Claude's conversation context after activation; later requests in that mode should go through the same `autoresearch agent --mode ... --prompt ...` bridge. Add `--wait` when the completed durable events and artifacts are needed in the current terminal response.

Review needs a discovered paper set. Novelty needs a validated gap. Experiment planning needs a novelty candidate. Experiment execution remains separately approval-gated in the backend and is never started by these skills.

# AutoResearch

An evidence-first research workflow engine: supply a seed paper, DOI, or URL
and it advances a durable pipeline — literature review → research question →
candidate hypotheses → versioned methodology → experimental design →
approval-gated execution → bounded revision — with every claim traceable to
verbatim-verified source text.

AutoResearch is for researchers who want literature grounding and falsifiable
designs produced *with provenance*, not polished prose without it. It does
**not** claim novelty, correctness, or publication quality; it records what
the evidence supports, what is uncertain, and what remains unverified.

## What it does (and does not do)

| Stage | What you get | Honest bound |
|---|---|---|
| Discovery + review | Papers from open scholarly APIs, legal-OA PDFs, evidence-backed reviews, one cited gap report | Only verified excerpts count as evidence; counter-evidence is preserved |
| Research question | One selected gap with selection rationale and classification (`evidence_supported` or `hypothesis`) | An absence in retrieved papers is never treated as proof of a gap |
| Hypotheses | A risk-tiered portfolio of candidates, each prior-art-checked and adversarially critiqued | Novelty is only ever "potentially novel" / "expert review required" |
| Methodology | A versioned, checksummed freeze: hypothesis, predicted observations, competing explanations, falsification conditions, algorithm outline, procedure, resource demands | Freezing records what will be tested; it certifies nothing |
| Design | A declarative experiment spec linked to the question and freeze, with baselines, ablations, seeds, and pre-registered interpretation rules | The planner never invents baseline numbers |
| Execution | Runs only after you approve an exact specification (checksum-bound); typed actions only, duplicate-dispatch protected | A completed run is reported as `executed_analysis_pending` until artifacts are ingested and analyzed — never as a result |
| Revision | Opt-in bounded cycles on revision-worthy decisions; prior attempts retained | Requires an analyzed report (see "Execution and results") |

## Installation

Requires Python 3.11+.

```bash
git clone <this-repository-url> autoresearch
cd autoresearch
python3 -m venv .venv && source .venv/bin/activate
pip install .
autoresearch version
```

Console scripts installed: `autoresearch` (CLI) and `autoresearch-api`
(local backend daemon).

**Where state lives:** research projects, runs, and artifacts are stored in
`./.autoresearch/` **relative to the directory you run commands from**. Run
later stages from the same directory you ran discovery in, or set
`AUTORESEARCH_DATA_ROOT` to a fixed path (full list of overrides:
`AUTORESEARCH_DATABASE_PATH`, `AUTORESEARCH_ARTIFACT_ROOT`,
`AUTORESEARCH_PROVIDER_CONFIG`, `AUTORESEARCH_SEARCH_SOURCES` — see
[docs/user_guide.md](docs/user_guide.md)).

## Provider configuration

Model-backed stages need a provider. Copy the template and export keys.
The configured environment variable takes precedence. If it is absent,
AutoResearch can read the matching API key from OpenCode's local credential
store for official DeepSeek, OpenAI, Anthropic, Google Gemini, and OpenRouter
endpoints. It never copies the key into AutoResearch state or logs, changes
the configured model/provider, or repurposes OAuth/subscription tokens.
Custom endpoints still require their configured environment variable.
See [credential handoff details](docs/claude_code.md#opencode-provider-credentials).

```bash
mkdir -p .autoresearch && cp .autoresearch/config.example.toml .autoresearch/config.toml
export AUTORESEARCH_LLM_API_KEY="sk-..."       # your provider key
export AUTORESEARCH_EMBEDDING_API_KEY="sk-..." # if using external embeddings
autoresearch skill doctor
```

Roles: `literature_extraction`, `research_reasoning`, `scientific_critic`
(all satisfied by one reasoning provider in the example), plus `embeddings`
for review ranking. Any OpenAI-compatible endpoint works; Anthropic and
Gemini wire formats are also supported. Without a provider, project/inspection
commands still work and model-backed stages report the missing role with a
remedy.

Verify role assignments with:

```bash
autoresearch skill doctor
```

Cost limits: `--max-cost-usd` sets a durable, project-scoped provider-cost
hard stop enforced by the router. It bounds model spend only — it is not an
execution-resource quota.

## First run (offline example, no API key)

```bash
autoresearch discover --from-catalog evals/fixtures/catalog_example.json
autoresearch review --project <project_id_from_output>
```

To import your own papers, build a catalog JSON (see
`evals/fixtures/catalog_example.json` for the exact shape) or supply a seed
paper. **Note:** review and later stages are model-backed and need provider
configuration; discovery + import work offline.

## The full cycle

```bash
# Stop after the literature review and question:
autoresearch workflow --seed ~/papers/seed.pdf --until question

# Resume to a frozen methodology and a linked experiment design:
autoresearch workflow --project <project_id> --until design

# Proceed to execution: prints the exact package (checksum, actions) and the
# missing requirement unless an approval already covers the spec:
autoresearch workflow --project <project_id> --until experiment
```

Every invocation prints per-stage dispositions (`completed` / `reused` /
`awaiting_approval` / …) with durable identities (`gap_id`,
`methodology_freeze_id`, `spec_id`, approval `checksum`). Re-invoking with
`--project` resumes: completed stages are reused, never repeated.

### Execution and results

Execution runs only through a policy-controlled worker with typed actions
(no arbitrary shell) and only after you approve the exact specification:

```bash
autoresearch skill experiment approve --candidate-id <id> --accept
autoresearch skill experiment execute --candidate-id <id>
```

**Current boundary (documented, not hidden):** after an approved run
completes, the workflow reports `executed_analysis_pending`. The step that
ingests measurement artifacts and runs the statistical analysis is not yet
wired into the CLI; until it is, `autoresearch skill experiment report`
correctly refuses with `insufficient_evidence` rather than inventing a
decision. You can inspect retained jobs and artifacts with
`autoresearch skill experiment inspect --candidate-id <id>` and
`autoresearch audit run <run_id>`.

### Resume and bounded revision

Completed stages and experiment identities survive interruption; duplicate
dispatch of an approved spec returns the same job. Revision is opt-in
(`--max-revisions N`), triggers only on revision-worthy analyzed decisions
(refuted / no-effect / insufficient-evidence / …), produces **new**
candidates, freezes, and specs while retaining prior attempts, and stops for
approval on every cycle.

## Agent skills (Claude Code / OpenCode)

Seven skills bridge the pipeline into agent hosts: `review`, `novelty`,
`experiment`, `reinvention`, `deep-research`, `workflow`, and the read-only
`status` snapshot.

**OpenCode behavior:** installed skills are *explicit-invocation-only* —
`opencode/autoinvoke: false` keeps them out of the model's auto-selection,
and each is exposed as a slash command (`/review`, `/novelty`,
`/experiment`, `/reinvention`, `/workflow`, `/deep-research`, `/status`).
They run only when you explicitly invoke them; they are never auto-suggested.

**Progress visibility:** `/status` (backed by `autoresearch skill status`)
is a read-only snapshot — active runs, current stage, and latest durable
artifacts per project. It is safe to invoke in any session while another
one is mid-run, so "is it still working?" is always one slash command away.

```bash
scripts/install_skills.sh                    # Claude Code (~/.claude/skills)
scripts/install_skills.sh --target opencode  # OpenCode (~/.config/opencode/skills)
scripts/install_skills.sh --target all       # both
autoresearch skill list                      # verify discovery + sync state
scripts/install_skills.sh --remove           # removes ONLY this tool's skills (all targets)
```

Skills invoke `${AUTORESEARCH_BIN:-autoresearch}`; make sure the CLI is on
`PATH` (or export `AUTORESEARCH_BIN`). Provider configuration is resolved
from `./.autoresearch/config.toml` or `$AUTORESEARCH_PROVIDER_CONFIG`.

The installer refuses to overwrite a skill directory it did not install
(use `--force` to override). Both hosts get the same canonical definitions
shipped inside the package.

## Verification status

- Deterministic offline test suite (clean checkout):
  `pip install -e ".[dev]" && pytest` — **115 passed / 1 skipped** at release;
  the single skip is the opt-in live-provider test
  (`RUN_LIVE_AUTORESEARCH_TESTS=1` enables paid/network verification).
  Maintainers additionally run an extended private suite covering the
  research workspace this tool was built for; those numbers are not part of
  this repository.
- The full-cycle demonstration (resume, approval gating, duplicate-dispatch,
  honest pending states) runs deterministically in
  `tests/test_workflow_coordination.py`.
- **Live provider quality is not evaluated here** — model-dependent research
  quality (hypothesis usefulness, design adequacy) requires your own
  provider and has not been benchmarked by the authors.

## Documentation

- [User guide](docs/user_guide.md) — CLI, workflow stages, controls
- [Architecture](docs/architecture.md) — system design and evidence guarantees
- [Security](docs/security.md) — credential handling
- [Eval suite](evals/README.md) — offline ResearchBench
- [Research track notes](docs/research_track.md) — the project this tool was built for

## License

MIT — see [LICENSE](LICENSE). You may use, modify, and redistribute this
software; research outputs it produces are yours and carry no novelty or
correctness guarantee (see "What it does (and does not do)").

# AutoResearch

An evidence-backed research cycle in five skills:

```
review → novelty → experiment → manuscript
        └───────── workflow (runs all four in a loop) ─────────┘
```

- **review** — discovers real literature (Semantic Scholar, OpenAlex,
  Crossref, arXiv, OpenReview), extracts verbatim-verified evidence, and
  surfaces a cited research question with counter-evidence.
- **novelty** — generates candidate methods grounded in the reviewed
  evidence, checks them against prior art, critiques them adversarially, and
  freezes the selected methodology (versioned, checksummed).
- **experiment** — turns the frozen methodology into working research code
  (generated + statically verified), plans a falsifiable spec, and executes
  it only under an explicit, exact-checksum approval. Measurements are
  ingested, identity-checked, recomputed from raw outputs where possible,
  and analyzed deterministically. Negative and null results are valid
  outcomes.
- **manuscript** — assembles the validated lineage into an editable,
  cited manuscript (Markdown + LaTeX + BibTeX + tables + manifest). Every
  number traces to a measurement artifact; missing evidence produces an
  explicitly incomplete draft, never a fabricated one.
- **workflow** — one durable loop over all four stages, reusing completed
  work and stopping where asked.

Every analyzed outcome is persisted to a shared **Findings Memory**
(including failures and refutations) that conditions later runs, so the
system does not repeatedly rediscover unsuccessful approaches.

## Quality width (parallel, latency-neutral)

Quality scales by *width, not depth*: the four skills add parallel fan-out
where coverage is the ceiling — more tokens, same wall-clock, no new serial
hops. All knobs default ON and degrade gracefully under partial failure;
tune them under `[research]` in `.autoresearch/config.toml` (the boolean and
perspective knobs also accept `AUTORESEARCH_*` env equivalents; the
anchors/neighbors bounds are config-file only):

- `discovery_query_perspectives` — parallel complementary perspectives for
  search-strategy generation (topic/method, application/domain,
  limitation/counter-evidence); the union is deduplicated.
- `discovery_citation_expansion` + `_anchors`/`_neighbors` — deterministic,
  zero-token OpenAlex citation chaining: forward/backward neighbors of the
  top-ranked candidates join the candidate pool and are re-ranked with it.
- `review_ensemble` — dual-leg parallel extraction per paper (second leg
  sees reversed chunk order), merged with claim deduplication; a failing
  leg degrades to the survivor.
- Prior-art checking (novelty) now fans out across **all** sources with six
  complementary queries per candidate, including cross-domain probes
  grounded in the seed's topic and datasets.

## Install

```bash
pip install -e .
```

## Configure

Inside an agent session, the skills automatically use the session's own AI
model and API key (auto-detected from the OpenCode credential store) — no
config file required. `AUTORESEARCH_AUTO_PROVIDER=true` or `[research]
auto_provider = true` forces session-following even when a config file
exists.

Otherwise copy `.autoresearch/config.example.toml` to
`.autoresearch/config.toml` and point it at your provider. Secrets are never
inlined: name an environment variable (`api_key_env`), or rely on ambient
resolution (OpenCode credential store for recognized endpoints,
`ANTHROPIC_API_KEY` / connector settings for Anthropic-protocol providers).

## Use

```bash
autoresearch skill workflow --seed-paper <pdf|doi|url> --until design
autoresearch skill experiment approve --candidate-id <id> --accept   # your approval gate
autoresearch skill workflow --project <project_id> --until manuscript

# or each skill separately:
autoresearch skill review     --project <id>
autoresearch skill novelty    --project <id>
autoresearch skill experiment plan --candidate-id <id>
autoresearch skill manuscript --project <id>
```

`autoresearch skill install --target opencode` (or `claude`) installs the
five skills into your agent host as explicit slash commands
(`/review`, `/novelty`, `/experiment`, `/manuscript`, `/workflow`).

## Honesty rules (enforced in code, not prompts)

- No fabricated citations, measurements, hardware facts, or outcomes — ever.
- Execution requires an explicit approval bound to the exact spec checksum.
- Fixture-labelled artifacts cannot pass scientific-evidence gates.
- Numerical authority is deterministic; model-assisted manuscript prose is
  audited (every number, citation, and overclaim checked) and falls back to
  the deterministic draft on any violation.
- An incomplete result stays visibly incomplete.

## Verify

```bash
pip install pytest anyio
pytest tests/ -q   # offline end-to-end smoke of the five-skill loop
```


## Where project state lives

Each research project stores its state **in its own directory** — the
directory you run the skills from when the project is created:

- The project's database, artifacts, and downloads live in
  `<project-dir>/.autoresearch/`.
- The project remembers that directory (exact database path recorded at
  creation, plus a machine-local registry at
  `~/.autoresearch/project-registry.json`).
- Resuming with `--project <id>` works **from any working directory** —
  the skills re-open the project's own database instead of starting a fresh
  one. Four projects in four folders stay four separate stores; nothing
  lands in the source checkout.

**Rule of thumb:** run `/workflow` (or the individual skills) from inside
the project's folder the first time; after that, resume from anywhere.

# First milestone report

## Delivered

- Seed parsing for local PDF, DOI, and HTTPS URL; DOI/URL seeds are hydrated from
  scholarly metadata before provider analysis.
- Provider-routed seed extraction and multiple search-strategy generation.
- Concurrent Semantic Scholar, OpenAlex, Crossref, arXiv, and OpenReview search.
- Metadata normalization and identifier-first deduplication.
- Transparent recent-paper ranking with complete search provenance.
- Legal PDF resolution through declared OA locations and Unpaywall, with
  explicit unavailable/rejected/failed states.
- Strict per-paper structured reviews with source-excerpt verification.
- Router-only external embedding ranking for top five.
- Evidence-constrained gap analysis, counter-evidence, and hypothesis labeling.
- Checksummed JSON and Markdown gap reports.
- Durable SQLite runs, failures, artifacts, provider usage, and budgets.
- Required `discover` and `review` CLI commands.

## Validation

Run from the repository root:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src python -m autoresearch.cli.app --help
python -m pip install -e '.[dev]'
python -m pip check
```

Latest verified result: **29 tests passed**, the editable package built in a
clean virtual environment, all CLI help commands executed, and `pip check`
reported no broken requirements. The deterministic end-to-end test
covers a seed PDF, four search strategies, 20 selected recent records, mixed
downloaded/unavailable PDFs, 20 evidence-backed reviews, top-five ranking, one
evidence-supported gap, both report formats, provider usage, completed durable
runs, and SQLite integrity. It uses no network or paid API key.

## Known operational limits

- No live provider or scholarly-source smoke test is run in CI because it would
  require credentials/network and could consume quotas.
- Open-access availability varies; 20 metadata records does not mean 20 PDFs.
- Scanned PDFs require a future OCR adapter.
- Very long PDFs are bounded before provider review; scaling deep review to 100
  papers requires hierarchical section synthesis and resume checkpoints.
- OpenReview coverage is best-effort and strongest for public conference notes.
- Novelty, experiments, reinvention, SSH execution, API, and frontend remain
  typed future boundaries only, exactly as scoped.

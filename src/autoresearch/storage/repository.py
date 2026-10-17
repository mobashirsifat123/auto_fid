"""SQLite repository for durable, provenance-preserving pipeline state."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    ArtifactReference,
    DownloadRecord,
    DownloadStatus,
    EvidenceClaim,
    FindingStatus,
    GapClassification,
    ManuscriptRecord,
    ResearchFinding,
    PaperRecord,
    PaperReview,
    PipelineKind,
    PipelineRun,
    PriorityScore,
    Project,
    ProviderUsage,
    ResearchGap,
    RunStatus,
    SearchCandidate,
    SeedPaper,
    utc_now,
    PriorArtStatus,
    CandidateStatus,
    ExperimentDecision,
    NoveltyCandidate,
    PriorArtAssessment,
    NoveltyCritique,
    CandidateRevision,
    NoveltyPortfolio,
    NoveltyClaimLedger,
    ExperimentSpec,
    EnvironmentManifest,
    ValidationReport,
    ExecutionJob,
    ExecutionResult,
    StatisticalAnalysis,
    ExperimentReport,
    ReinventionAction,
    ReinventionAnalysis,
)


SCHEMA_VERSION = 16


def _register_project_root(project_id: str, data_root: str) -> None:
    """Machine-local registry: project id -> the directory its state lives
    in. Lets any later command anchor to the project from an arbitrary
    working directory (resume never starts a fresh database by accident)."""
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    registry_path = _Path(_os.environ.get(
        "AUTORESEARCH_REGISTRY_PATH", str(_Path.home() / ".autoresearch" / "project-registry.json"),
    )).expanduser()
    try:
        payload: dict = {}
        if registry_path.is_file():
            try:
                payload = _json.loads(registry_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            except _json.JSONDecodeError:
                payload = {}
        payload[str(project_id)] = str(data_root)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = registry_path.with_suffix(".tmp")
        temporary.write_text(_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(registry_path)
    except OSError:
        # Registry writes are best-effort bookkeeping; the project row itself
        # still carries data_root for same-database lookups.
        pass


class ResearchRepository:
    def __init__(self, path: Path, artifact_root: Path) -> None:
        self.path = path.expanduser().resolve()
        self.artifact_root = artifact_root.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            current = int(connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0])
            if current < 1:
                connection.executescript(_SCHEMA_V1)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (1, utc_now()))
            if current < 2:
                connection.executescript(_SCHEMA_V2)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (2, utc_now()))
            if current < 3:
                connection.executescript(_SCHEMA_V3)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (3, utc_now()))
            if current < 4:
                connection.executescript(_SCHEMA_V4)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (4, utc_now()))
            if current < 5:
                connection.executescript(_SCHEMA_V5)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (5, utc_now()))
            if current < 6:
                connection.executescript(_SCHEMA_V6)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (6, utc_now()))
            if current < 7:
                connection.executescript(_SCHEMA_V7)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (7, utc_now()))
            if current < 8:
                connection.executescript(_SCHEMA_V8)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (8, utc_now()))
            if current < 9:
                connection.executescript(_SCHEMA_V9)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (9, utc_now()))
            if current < 10:
                connection.executescript(_SCHEMA_V10)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (10, utc_now()))
            if current < 11:
                connection.executescript(_SCHEMA_V11)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (11, utc_now()))
            if current < 12:
                connection.executescript(_SCHEMA_V12)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (12, utc_now()))
            if current < 13:
                connection.executescript(_SCHEMA_V13)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (13, utc_now()))
            if current < 14:
                connection.executescript(_SCHEMA_V14)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (14, utc_now()))
            if current < 15:
                connection.executescript(_SCHEMA_V15)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (15, utc_now()))
            if current < 16:
                connection.executescript(_SCHEMA_V16)
                connection.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)", (16, utc_now()))
            latest = int(connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0])
            if latest != SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported database schema v{latest}; expected v{SCHEMA_VERSION}.")

    def create_project(self, title: str) -> Project:
        project = Project(uuid4().hex, title.strip())
        if not project.title:
            raise ValueError("Project title is required.")
        # The project remembers where its state lives: later commands can
        # anchor to this directory from any CWD instead of creating a fresh
        # database wherever they happen to run.
        state_db_path = str(self.path.resolve())
        self.execute("INSERT INTO projects(id,title,created_at,state_db_path) VALUES(?,?,?,?)", (project.id, project.title, project.created_at, state_db_path))
        _register_project_root(project.id, state_db_path)
        return project

    def get_project(self, project_id: str) -> Project | None:
        row = self.row("SELECT * FROM projects WHERE id=?", (project_id,))
        return Project(**row) if row else None

    def link_project(self, source_project_id: str, target_project_id: str) -> None:
        """Link papers from a source project into a target project (for cross-project references)."""
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO project_links(source_project_id, target_project_id, created_at) VALUES(?,?,?)",
                (source_project_id, target_project_id, utc_now()),
            )
            connection.execute(
                """INSERT OR IGNORE INTO papers
                   (id,project_id,run_id,title,abstract,authors_json,year,publication_date,doi,arxiv_id,url,source_names_json,open_access_pdf_url,is_open_access,citation_count,venue,checksum,created_at)
                   SELECT id,?,run_id,title,abstract,authors_json,year,publication_date,doi,arxiv_id,url,source_names_json,open_access_pdf_url,is_open_access,citation_count,venue,checksum,created_at
                   FROM papers WHERE project_id=?""",
                (target_project_id, source_project_id),
            )

    def list_projects(self) -> list[Project]:
        return [Project(**row) for row in self.rows("SELECT * FROM projects ORDER BY created_at DESC")]

    def create_run(self, project_id: str, kind: PipelineKind, input_data: dict[str, Any]) -> PipelineRun:
        run = PipelineRun(uuid4().hex, project_id, kind, RunStatus.RUNNING, input_data)
        self.execute(
            "INSERT INTO pipeline_runs(id,project_id,kind,status,input_json,started_at) VALUES(?,?,?,?,?,?)",
            (run.id, project_id, kind.value, run.status.value, _json(input_data), run.started_at),
        )
        return run

    def finish_run(self, run_id: str, *, error: str | None = None) -> None:
        status = RunStatus.FAILED if error else RunStatus.COMPLETED
        self.execute("UPDATE pipeline_runs SET status=?,completed_at=?,error=? WHERE id=?", (status.value, utc_now(), error, run_id))

    def resume_run(self, run_id: str) -> PipelineRun:
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' was not found.")
        self.execute(
            "UPDATE pipeline_runs SET status=?,completed_at=NULL,error=NULL WHERE id=?",
            (RunStatus.RUNNING.value, run_id),
        )
        resumed = self.get_run(run_id)
        assert resumed is not None
        return resumed

    def get_run(self, run_id: str) -> PipelineRun | None:
        row = self.row("SELECT * FROM pipeline_runs WHERE id=?", (run_id,))
        if not row:
            return None
        return PipelineRun(row["id"], row["project_id"], PipelineKind(row["kind"]), RunStatus(row["status"]), _loads(row["input_json"], {}), row["started_at"], row["completed_at"], row["error"])

    def list_runs(self, *, project_id: str | None = None, status: RunStatus | None = None) -> list[PipelineRun]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if status:
            clauses.append("status=?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.rows(f"SELECT id FROM pipeline_runs{where} ORDER BY started_at DESC", tuple(params))
        result: list[PipelineRun] = []
        for row in rows:
            run = self.get_run(row["id"])
            if run:
                result.append(run)
        return result

    def cancel_run(self, run_id: str) -> None:
        self.execute("UPDATE pipeline_runs SET status=?,completed_at=?,error=? WHERE id=? AND status IN (?,?)", (RunStatus.CANCELLED.value, utc_now(), "Cancelled by user.", run_id, RunStatus.RUNNING.value, RunStatus.PENDING.value))

    def latest_run(self, project_id: str, kind: PipelineKind) -> PipelineRun | None:
        row = self.row(
            "SELECT id FROM pipeline_runs WHERE project_id=? AND kind=? ORDER BY started_at DESC LIMIT 1",
            (project_id, kind.value),
        )
        return self.get_run(row["id"]) if row else None

    def begin_checkpoint(self, run_id: str, stage: str, item_key: str = "main") -> None:
        self.execute(
            """INSERT INTO pipeline_checkpoints(run_id,stage,item_key,status,payload_json,attempts,error,updated_at)
               VALUES(?,?,?,'running',NULL,1,NULL,?)
               ON CONFLICT(run_id,stage,item_key) DO UPDATE SET
                 status='running',attempts=pipeline_checkpoints.attempts+1,error=NULL,updated_at=excluded.updated_at""",
            (run_id, stage, item_key, utc_now()),
        )

    def complete_checkpoint(self, run_id: str, stage: str, item_key: str = "main", *, payload: Any = None) -> None:
        self.execute(
            """INSERT INTO pipeline_checkpoints(run_id,stage,item_key,status,payload_json,attempts,error,updated_at)
               VALUES(?,?,?,'completed',?,1,NULL,?)
               ON CONFLICT(run_id,stage,item_key) DO UPDATE SET
                 status='completed',payload_json=excluded.payload_json,error=NULL,updated_at=excluded.updated_at""",
            (run_id, stage, item_key, _json(payload) if payload is not None else None, utc_now()),
        )

    def fail_checkpoint(self, run_id: str, stage: str, item_key: str = "main", *, error: str) -> None:
        self.execute(
            """INSERT INTO pipeline_checkpoints(run_id,stage,item_key,status,payload_json,attempts,error,updated_at)
               VALUES(?,?,?,'failed',NULL,1,?,?)
               ON CONFLICT(run_id,stage,item_key) DO UPDATE SET
                 status='failed',error=excluded.error,updated_at=excluded.updated_at""",
            (run_id, stage, item_key, error, utc_now()),
        )

    def checkpoint(self, run_id: str, stage: str, item_key: str = "main") -> dict[str, Any] | None:
        row = self.row(
            "SELECT * FROM pipeline_checkpoints WHERE run_id=? AND stage=? AND item_key=?",
            (run_id, stage, item_key),
        )
        if row and row.get("payload_json"):
            row["payload"] = _loads(row["payload_json"], None)
        return row

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.rows(
            "SELECT * FROM pipeline_checkpoints WHERE run_id=? ORDER BY stage,item_key",
            (run_id,),
        )
        for row in rows:
            row["payload"] = _loads(row.get("payload_json"), None)
        return rows

    def save_seed(self, seed: SeedPaper) -> None:
        self.execute(
            """INSERT OR REPLACE INTO seed_papers
               (id,project_id,run_id,input_type,input_value,title,text,topic,research_question,doi,url,methods_json,datasets_json,citations_json,limitations_json,checksum,provenance_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (seed.id, seed.project_id, seed.run_id, seed.input_type, seed.input_value, seed.title, seed.text, seed.topic, seed.research_question, seed.doi, seed.url, _json(seed.methods), _json(seed.datasets), _json(seed.citations), _json(seed.limitations), seed.checksum, _json_provenance(seed.provenance), seed.created_at),
        )

    def get_seed(self, project_id: str) -> SeedPaper | None:
        row = self.row("SELECT * FROM seed_papers WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
        if not row:
            return None
        return SeedPaper(
            id=row["id"], project_id=row["project_id"], run_id=row["run_id"], input_type=row["input_type"], input_value=row["input_value"],
            title=row["title"], text=row["text"], topic=row["topic"], research_question=row["research_question"], checksum=row["checksum"], provenance=_provenance(row["provenance_json"]),
            doi=row["doi"], url=row["url"], methods=tuple(_loads(row["methods_json"], [])), datasets=tuple(_loads(row["datasets_json"], [])), citations=tuple(_loads(row["citations_json"], [])), limitations=tuple(_loads(row["limitations_json"], [])), created_at=row["created_at"],
        )

    def record_search_event(self, *, project_id: str, run_id: str, source: str, strategy: str, query: str, requested_limit: int, result_count: int, raw_checksum: str, error: str | None = None) -> str:
        identifier = uuid4().hex
        self.execute("INSERT INTO search_events(id,project_id,run_id,source,strategy,query,requested_limit,result_count,raw_checksum,error,retrieved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (identifier, project_id, run_id, source, strategy, query, requested_limit, result_count, raw_checksum, error, utc_now()))
        return identifier

    def search_events(self, run_id: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM search_events WHERE run_id=? ORDER BY source,strategy", (run_id,))

    def save_paper(self, paper: PaperRecord) -> None:
        self.execute(
            """INSERT INTO papers
               (id,project_id,run_id,title,abstract,authors_json,year,publication_date,doi,arxiv_id,url,source_names_json,open_access_pdf_url,is_open_access,citation_count,venue,checksum,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title,abstract=excluded.abstract,authors_json=excluded.authors_json,year=excluded.year,
                 publication_date=excluded.publication_date,doi=COALESCE(excluded.doi,papers.doi),arxiv_id=COALESCE(excluded.arxiv_id,papers.arxiv_id),
                 url=COALESCE(excluded.url,papers.url),source_names_json=excluded.source_names_json,
                 open_access_pdf_url=COALESCE(excluded.open_access_pdf_url,papers.open_access_pdf_url),is_open_access=MAX(papers.is_open_access,excluded.is_open_access),
                 citation_count=MAX(papers.citation_count,excluded.citation_count),venue=COALESCE(NULLIF(excluded.venue,''),papers.venue),checksum=excluded.checksum""",
            (paper.id, paper.project_id, paper.run_id, paper.title, paper.abstract, _json(paper.authors), paper.year, paper.publication_date, paper.doi, paper.arxiv_id, paper.url, _json(paper.source_names), paper.open_access_pdf_url, int(paper.is_open_access), paper.citation_count, paper.venue, paper.checksum, paper.created_at),
        )

    def import_papers_batch(self, papers: list[PaperRecord]) -> int:
        """Import a batch of papers from a local catalog, skipping duplicates."""
        if not papers:
            return 0
        with self._lock, self.connect() as connection:
            count = 0
            for paper in papers:
                existing = connection.execute("SELECT 1 FROM papers WHERE id=?", (paper.id,)).fetchone()
                if existing:
                    continue
                connection.execute(
                    """INSERT INTO papers
                       (id,project_id,run_id,title,abstract,authors_json,year,publication_date,doi,arxiv_id,url,source_names_json,open_access_pdf_url,is_open_access,citation_count,venue,checksum,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (paper.id, paper.project_id, paper.run_id, paper.title, paper.abstract, _json(paper.authors), paper.year, paper.publication_date, paper.doi, paper.arxiv_id, paper.url, _json(paper.source_names), paper.open_access_pdf_url, int(paper.is_open_access), paper.citation_count, paper.venue, paper.checksum, paper.created_at),
                )
                count += 1
            return count

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        row = self.row("SELECT * FROM papers WHERE id=?", (paper_id,))
        return _paper(row) if row else None

    def list_papers(self, project_id: str) -> list[PaperRecord]:
        return [_paper(row) for row in self.rows("SELECT * FROM papers WHERE project_id=? ORDER BY publication_date DESC,title", (project_id,))]

    def save_candidate(self, candidate: SearchCandidate) -> None:
        self.execute("INSERT OR IGNORE INTO search_candidates(id,project_id,run_id,paper_id,source,strategy,source_rank,query,raw_checksum,retrieved_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (candidate.id, candidate.project_id, candidate.run_id, candidate.paper_id, candidate.source, candidate.strategy, candidate.source_rank, candidate.query, candidate.raw_checksum, candidate.retrieved_at))

    def save_discovery_score(self, *, project_id: str, run_id: str, paper_id: str, components: dict[str, float]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO discovery_scores(project_id,run_id,paper_id,components_json,total,created_at) VALUES(?,?,?,?,?,?)",
            (project_id, run_id, paper_id, _json(components), float(components["total"]), utc_now()),
        )

    def discovery_scores(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.rows("SELECT * FROM discovery_scores WHERE run_id=? ORDER BY total DESC,paper_id", (run_id,))
        for row in rows:
            row["components"] = _loads(row.pop("components_json"), {})
        return rows

    def save_download(self, record: DownloadRecord) -> None:
        self.execute("INSERT OR REPLACE INTO download_records(id,project_id,run_id,paper_id,status,source,url,legal_basis,license,path,reason,checksum,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (record.id, record.project_id, record.run_id, record.paper_id, record.status.value, record.source, record.url, record.legal_basis, record.license, record.path, record.reason, record.checksum, record.created_at))

    def downloads(self, project_id: str) -> list[DownloadRecord]:
        return [_download(row) for row in self.rows("SELECT * FROM download_records WHERE project_id=? ORDER BY created_at", (project_id,))]

    def download_for_paper(self, paper_id: str) -> DownloadRecord | None:
        row = self.row("SELECT * FROM download_records WHERE paper_id=? ORDER BY created_at DESC LIMIT 1", (paper_id,))
        return _download(row) if row else None

    def save_review(self, review: PaperReview) -> None:
        self.execute(
            """INSERT OR REPLACE INTO paper_reviews
               (id,project_id,run_id,paper_id,problem,research_question,method,datasets_json,experimental_protocol,results_json,ablations_json,limitations_json,failure_modes_json,evidence_claim_ids_json,confidence,provenance_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (review.id, review.project_id, review.run_id, review.paper_id, review.problem, review.research_question, review.method, _json(review.datasets), review.experimental_protocol, _json(review.results), _json(review.ablations), _json(review.limitations), _json(review.failure_modes), _json(review.evidence_claim_ids), review.confidence, _json_provenance(review.provenance)),
        )

    def record_review_failure(self, *, project_id: str, run_id: str, paper_id: str, reason: str) -> None:
        self.execute("INSERT INTO review_failures(id,project_id,run_id,paper_id,reason,created_at) VALUES(?,?,?,?,?,?)", (uuid4().hex, project_id, run_id, paper_id, reason, utc_now()))

    def clear_review_failures(self, run_id: str, paper_id: str) -> None:
        self.execute("DELETE FROM review_failures WHERE run_id=? AND paper_id=?", (run_id, paper_id))

    def review_failures(self, run_id: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM review_failures WHERE run_id=? ORDER BY paper_id", (run_id,))

    def reviews(self, project_id: str, run_id: str | None = None) -> list[PaperReview]:
        sql = "SELECT * FROM paper_reviews WHERE project_id=?"
        params: tuple[Any, ...] = (project_id,)
        if run_id:
            sql += " AND run_id=?"
            params = (project_id, run_id)
        return [_review(row) for row in self.rows(sql + " ORDER BY paper_id", params)]

    def latest_reusable_review(self, paper: "PaperRecord", *, hierarchical: bool) -> PaperReview | None:
        """Latest completed paper checkpoint from any run of the same
        project whose prompt version matches the requested review mode.
        Reviews are verbatim-verified evidence for an immutable paper under a
        fixed prompt version, so reuse is sound; a version bump or --fresh
        forces re-extraction."""
        version_prefix = "paper-review-hierarchical-v1" if hierarchical else "paper-review-v2"
        row = self.row(
            "SELECT pr.* FROM paper_reviews pr"
            " JOIN pipeline_checkpoints cp ON cp.run_id=pr.run_id"
            " AND cp.stage='paper_review' AND cp.item_key=pr.paper_id AND cp.status='completed'"
            " WHERE pr.project_id = ? AND pr.paper_id = ?"
            "   AND pr.provenance_json LIKE ?"
            " ORDER BY pr.rowid DESC LIMIT 1",
            (paper.project_id, paper.id, f'%"prompt_version": "{version_prefix}%'),
        )
        return _review(row) if row else None

    def review_for_paper(self, run_id: str, paper_id: str) -> PaperReview | None:
        row = self.row(
            "SELECT * FROM paper_reviews WHERE run_id=? AND paper_id=?",
            (run_id, paper_id),
        )
        return _review(row) if row else None

    def save_claim(self, claim: EvidenceClaim) -> None:
        self.execute("INSERT OR REPLACE INTO evidence_claims(id,project_id,run_id,paper_id,claim_type,statement,excerpt,page,section,confidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (claim.id, claim.project_id, claim.run_id, claim.paper_id, claim.claim_type, claim.statement, claim.excerpt, claim.page, claim.section, claim.confidence, _json_provenance(claim.provenance)))

    def claims(self, project_id: str, *, ids: tuple[str, ...] | None = None) -> list[EvidenceClaim]:
        rows = self.rows("SELECT * FROM evidence_claims WHERE project_id=? ORDER BY id", (project_id,))
        claims = [_claim(row) for row in rows]
        return [claim for claim in claims if not ids or claim.id in ids]

    def save_score(self, score: PriorityScore) -> None:
        self.execute("INSERT OR REPLACE INTO priority_scores(id,project_id,run_id,paper_id,semantic_similarity,methodological_similarity,research_question_similarity,recency,influence,evidence_quality,total,explanation_json,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (score.id, score.project_id, score.run_id, score.paper_id, score.semantic_similarity, score.methodological_similarity, score.research_question_similarity, score.recency, score.influence, score.evidence_quality, score.total, _json(score.explanation), _json_provenance(score.provenance)))

    def scores(self, project_id: str, run_id: str | None = None) -> list[PriorityScore]:
        sql = "SELECT * FROM priority_scores WHERE project_id=?"
        params: tuple[Any, ...] = (project_id,)
        if run_id:
            sql += " AND run_id=?"
            params = (project_id, run_id)
        return [_score(row) for row in self.rows(sql + " ORDER BY total DESC,paper_id", params)]

    def save_gap(self, gap: ResearchGap) -> None:
        self.execute("INSERT OR REPLACE INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (gap.id, gap.project_id, gap.run_id, gap.statement, _json(gap.supporting_paper_ids), _json(gap.supporting_claim_ids), gap.why_unsolved, gap.missing_experiment, gap.falsification_test, gap.confidence, _json(gap.counter_evidence_claim_ids), gap.classification.value, _json_provenance(gap.provenance)))

    def latest_gap(self, project_id: str) -> ResearchGap | None:
        row = self.row("SELECT * FROM research_gaps WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project_id,))
        return _gap(row) if row else None

    def gap_for_run(self, run_id: str) -> ResearchGap | None:
        row = self.row("SELECT * FROM research_gaps WHERE run_id=? ORDER BY rowid DESC LIMIT 1", (run_id,))
        return _gap(row) if row else None

    def save_artifact(self, artifact: ArtifactReference) -> None:
        self.execute("INSERT OR REPLACE INTO artifacts(id,project_id,run_id,artifact_type,path,source_evidence_json,provider,model,prompt_version,checksum,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (artifact.id, artifact.project_id, artifact.run_id, artifact.artifact_type, artifact.path, _json(artifact.source_evidence), artifact.provider, artifact.model, artifact.prompt_version, artifact.checksum, artifact.created_at))

    def artifacts(self, project_id: str) -> list[ArtifactReference]:
        return [ArtifactReference(row["id"], row["project_id"], row["run_id"], row["artifact_type"], row["path"], tuple(_loads(row["source_evidence_json"], [])), row["provider"], row["model"], row["prompt_version"], row["checksum"], row["created_at"]) for row in self.rows("SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at", (project_id,))]

    def artifacts_for_run(self, run_id: str) -> list[ArtifactReference]:
        rows = self.rows("SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at", (run_id,))
        return [ArtifactReference(row["id"], row["project_id"], row["run_id"], row["artifact_type"], row["path"], tuple(_loads(row["source_evidence_json"], [])), row["provider"], row["model"], row["prompt_version"], row["checksum"], row["created_at"]) for row in rows]

    def record_provider_usage(self, usage: ProviderUsage) -> None:
        self.execute("INSERT INTO provider_usage(id,project_id,run_id,operation,role,provider,model,prompt_tokens,completion_tokens,estimated_cost,currency,request_id,latency_ms,session_id,prompt_version,retry_count,structured_output_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (usage.id, usage.project_id, usage.run_id, usage.operation, usage.role, usage.provider, usage.model, usage.prompt_tokens, usage.completion_tokens, usage.estimated_cost, usage.currency, usage.request_id, usage.latency_ms, usage.session_id, usage.prompt_version, usage.retry_count, usage.structured_output_status, usage.created_at))

    def provider_spend(self, project_id: str) -> float:
        row = self.row("SELECT COALESCE(SUM(estimated_cost),0) AS total FROM provider_usage WHERE project_id=?", (project_id,))
        return float(row["total"] if row else 0.0)

    def provider_tokens(self, project_id: str) -> int:
        row = self.row(
            "SELECT COALESCE(SUM(COALESCE(prompt_tokens,0)+COALESCE(completion_tokens,0)),0) AS tokens"
            " FROM provider_usage WHERE project_id=?",
            (project_id,),
        )
        return int(row["tokens"] if row else 0)

    def set_project_budget(self, project_id: str, *, hard_stop: float | None = None, currency: str = "USD", token_ceiling: int | None = None) -> None:
        """Set the durable project budget. A ``None`` value PRESERVES the
        existing setting (so a cost-only invocation does not wipe a token
        ceiling, and vice versa); pass an explicit value to change it."""
        existing = self.get_project_budget(project_id) or {}
        effective_hard_stop = hard_stop if hard_stop is not None else existing.get("hard_stop")
        effective_ceiling = token_ceiling if token_ceiling is not None else existing.get("token_ceiling")
        self.execute(
            "INSERT INTO project_budgets(project_id,hard_stop,currency,token_ceiling,updated_at) VALUES(?,?,?,?,?)"
            " ON CONFLICT(project_id) DO UPDATE SET hard_stop=excluded.hard_stop,currency=excluded.currency,token_ceiling=excluded.token_ceiling,updated_at=excluded.updated_at",
            (project_id, effective_hard_stop, currency, effective_ceiling, utc_now()),
        )

    def get_project_budget(self, project_id: str) -> dict[str, Any] | None:
        return self.row("SELECT * FROM project_budgets WHERE project_id=?", (project_id,))

    def usage(self, *, project_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.rows(f"SELECT * FROM provider_usage{where} ORDER BY created_at", tuple(params))

    def append_api_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self._lock, self.connect() as connection:
            cursor = connection.execute("INSERT INTO api_events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)", (run_id, event_type, _json(payload), utc_now()))
            return int(cursor.lastrowid)

    def api_events(self, run_id: str, *, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.rows("SELECT id,run_id,event_type,payload_json,created_at FROM api_events WHERE run_id=? AND id>? ORDER BY id LIMIT ?", (run_id, after_id, limit))
        for row in rows:
            row["payload"] = _loads(row.pop("payload_json"), {})
        return rows

    def idempotency(self, key: str) -> dict[str, Any] | None:
        row = self.row("SELECT * FROM api_idempotency WHERE key=?", (key,))
        if row:
            row["response"] = _loads(row.pop("response_json"), {})
        return row

    def save_idempotency(self, key: str, operation: str, run_id: str, response: dict[str, Any]) -> None:
        self.execute("INSERT OR IGNORE INTO api_idempotency(key,operation,run_id,response_json,created_at) VALUES(?,?,?,?,?)", (key, operation, run_id, _json(response), utc_now()))

    def project_idempotency(self, key: str) -> dict[str, Any] | None:
        row = self.row("SELECT * FROM api_project_idempotency WHERE key=?", (key,))
        if row:
            row["response"] = _loads(row.pop("response_json"), {})
        return row

    def save_project_idempotency(self, key: str, project_id: str, response: dict[str, Any]) -> None:
        self.execute("INSERT OR IGNORE INTO api_project_idempotency(key,project_id,response_json,created_at) VALUES(?,?,?,?)", (key, project_id, _json(response), utc_now()))

    def approval(self, run_id: str) -> dict[str, Any] | None:
        return self.row("SELECT * FROM run_approvals WHERE run_id=?", (run_id,))

    def set_approval(self, run_id: str, status: str, reason: str | None = None) -> None:
        self.execute("INSERT INTO run_approvals(run_id,status,reason,updated_at) VALUES(?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,reason=excluded.reason,updated_at=excluded.updated_at", (run_id, status, reason, utc_now()))

    def integrity_check(self) -> str:
        row = self.row("PRAGMA integrity_check")
        return str(next(iter(row.values()))) if row else "unknown"

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(sql, params)

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock, self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def row(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.rows(sql, params)
        return rows[0] if rows else None

    # Novelty methods
    def save_novelty_candidate(self, candidate: NoveltyCandidate) -> None:
        self.execute(
            "INSERT INTO novelty_candidates(id,project_id,run_id,gap_id,strategy,method,causal_mechanism,target_limitation,difference_from_existing,expected_effect,falsification_condition,minimum_validating_experiment,required_ablations_json,implementation_complexity,expected_compute,risks_and_objections_json,status,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (candidate.id, candidate.project_id, candidate.run_id, candidate.gap_id, candidate.strategy, candidate.method, candidate.causal_mechanism, candidate.target_limitation, candidate.difference_from_existing, candidate.expected_effect, candidate.falsification_condition, candidate.minimum_validating_experiment, _json(candidate.required_ablations), candidate.implementation_complexity, candidate.expected_compute, _json(candidate.risks_and_objections), candidate.status.value, _json_provenance(candidate.provenance))
        )

    def update_novelty_candidate_status(self, candidate_id: str, status: CandidateStatus) -> None:
        self.execute("UPDATE novelty_candidates SET status=? WHERE id=?", (status.value, candidate_id))

    def novelty_candidates(self, run_id: str) -> list[NoveltyCandidate]:
        return [_novelty_candidate(row) for row in self.rows("SELECT * FROM novelty_candidates WHERE run_id=?", (run_id,))]

    def get_novelty_candidate(self, candidate_id: str) -> NoveltyCandidate | None:
        row = self.row("SELECT * FROM novelty_candidates WHERE id=?", (candidate_id,))
        return _novelty_candidate(row) if row else None

    def save_prior_art_assessment(self, assessment: PriorArtAssessment) -> None:
        self.execute(
            "INSERT INTO prior_art_assessments(id,project_id,run_id,candidate_id,closest_paper_ids_json,overlap_status,mechanism_comparison,supporting_excerpts_json,unresolved_overlap,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (assessment.id, assessment.project_id, assessment.run_id, assessment.candidate_id, _json(assessment.closest_paper_ids), assessment.overlap_status.value, assessment.mechanism_comparison, _json(assessment.supporting_excerpts), assessment.unresolved_overlap, _json_provenance(assessment.provenance))
        )

    def get_prior_art_assessment(self, candidate_id: str) -> PriorArtAssessment | None:
        row = self.row("SELECT * FROM prior_art_assessments WHERE candidate_id=?", (candidate_id,))
        return _prior_art_assessment(row) if row else None

    def save_novelty_critique(self, critique: NoveltyCritique) -> None:
        self.execute(
            "INSERT INTO novelty_critiques(id,project_id,run_id,candidate_id,novelty_score,significance_score,causal_plausibility_score,falsifiability_score,experimental_discriminability_score,feasibility_score,implementation_risk_score,expected_information_gain_score,compute_efficiency_score,publication_potential_score,objections_json,failure_scenarios_json,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (critique.id, critique.project_id, critique.run_id, critique.candidate_id, critique.novelty_score, critique.significance_score, critique.causal_plausibility_score, critique.falsifiability_score, critique.experimental_discriminability_score, critique.feasibility_score, critique.implementation_risk_score, critique.expected_information_gain_score, critique.compute_efficiency_score, critique.publication_potential_score, _json(critique.objections), _json(critique.failure_scenarios), _json_provenance(critique.provenance))
        )

    def get_novelty_critique(self, candidate_id: str) -> NoveltyCritique | None:
        row = self.row("SELECT * FROM novelty_critiques WHERE candidate_id=?", (candidate_id,))
        return _novelty_critique(row) if row else None

    def save_candidate_revision(self, revision: CandidateRevision) -> None:
        self.execute(
            "INSERT INTO candidate_revisions(id,project_id,run_id,original_candidate_id,revised_candidate_id,critique_id,revision_count,changes_made,provenance_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (revision.id, revision.project_id, revision.run_id, revision.original_candidate_id, revision.revised_candidate_id, revision.critique_id, revision.revision_count, revision.changes_made, _json_provenance(revision.provenance))
        )

    def candidate_revisions(self, original_candidate_id: str) -> list[CandidateRevision]:
        return [_candidate_revision(row) for row in self.rows("SELECT * FROM candidate_revisions WHERE original_candidate_id=? ORDER BY revision_count", (original_candidate_id,))]

    def save_novelty_portfolio(self, portfolio: NoveltyPortfolio) -> None:
        self.execute("INSERT OR REPLACE INTO novelty_portfolio_records(id,project_id,run_id,selection_reasons_json,provenance_json) VALUES(?,?,?,?,?)",
            (portfolio.id, portfolio.project_id, portfolio.run_id, _json(portfolio.selection_reasons), _json_provenance(portfolio.provenance)))
        for tier, candidate_id in (("low_risk", portfolio.low_risk_candidate_id), ("balanced", portfolio.balanced_candidate_id), ("high_risk", portfolio.high_risk_candidate_id)):
            if candidate_id:
                self.execute("INSERT OR REPLACE INTO novelty_portfolio_members(portfolio_id,tier,candidate_id) VALUES(?,?,?)", (portfolio.id, tier, candidate_id))
        if portfolio.low_risk_candidate_id and portfolio.balanced_candidate_id and portfolio.high_risk_candidate_id:
            self.execute(
                "INSERT OR REPLACE INTO novelty_portfolios(id,project_id,run_id,low_risk_candidate_id,balanced_candidate_id,high_risk_candidate_id,selection_reasons_json,provenance_json) VALUES(?,?,?,?,?,?,?,?)",
                (portfolio.id, portfolio.project_id, portfolio.run_id, portfolio.low_risk_candidate_id, portfolio.balanced_candidate_id, portfolio.high_risk_candidate_id, _json(portfolio.selection_reasons), _json_provenance(portfolio.provenance))
            )

    def _portfolio_from_record(self, record: dict[str, Any]) -> NoveltyPortfolio:
        members = {row["tier"]: row["candidate_id"] for row in self.rows("SELECT tier,candidate_id FROM novelty_portfolio_members WHERE portfolio_id=?", (record["id"],))}
        return NoveltyPortfolio(record["id"], record["project_id"], record["run_id"], members.get("low_risk"), members.get("balanced"), members.get("high_risk"), _loads(record["selection_reasons_json"], {}), _provenance(record["provenance_json"]))

    def get_novelty_portfolio(self, run_id: str) -> NoveltyPortfolio | None:
        record = self.row("SELECT * FROM novelty_portfolio_records WHERE run_id=?", (run_id,))
        if record:
            return self._portfolio_from_record(record)
        row = self.row("SELECT * FROM novelty_portfolios WHERE run_id=?", (run_id,))
        return _novelty_portfolio(row) if row else None

    def get_novelty_portfolio_by_id(self, portfolio_id: str) -> NoveltyPortfolio | None:
        record = self.row("SELECT * FROM novelty_portfolio_records WHERE id=?", (portfolio_id,))
        if record:
            return self._portfolio_from_record(record)
        row = self.row("SELECT * FROM novelty_portfolios WHERE id=?", (portfolio_id,))
        return _novelty_portfolio(row) if row else None

    def novelty_portfolios_for_project(self, project_id: str) -> list[NoveltyPortfolio]:
        # Newest first. The record schema is authoritative; legacy rows are
        # appended for back-compat. Three-tier portfolios are written to BOTH
        # tables, so dedupe by id before returning.
        seen: set[str] = set()
        portfolios: list[NoveltyPortfolio] = []
        for record in self.rows("SELECT * FROM novelty_portfolio_records WHERE project_id=? ORDER BY rowid DESC", (project_id,)):
            portfolios.append(self._portfolio_from_record(record))
            seen.add(str(record["id"]))
        for row in self.rows("SELECT * FROM novelty_portfolios WHERE project_id=? ORDER BY rowid DESC", (project_id,)):
            if str(row["id"]) not in seen:
                portfolios.append(_novelty_portfolio(row))
        return portfolios

    def latest_novelty_portfolio(self, project_id: str) -> NoveltyPortfolio | None:
        portfolios = self.novelty_portfolios_for_project(project_id)
        return portfolios[0] if portfolios else None

    def save_novelty_claim_ledger(self, ledger: NoveltyClaimLedger) -> None:
        self.execute(
            "INSERT OR REPLACE INTO novelty_claim_ledgers(id,project_id,run_id,candidate_id,contribution_claim,closest_methods_json,overlap_status,supporting_evidence_json,unresolved_uncertainty,safe_wording,prohibited_wording_json,review_status,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ledger.id, ledger.project_id, ledger.run_id, ledger.candidate_id, ledger.contribution_claim, _json(ledger.closest_methods), ledger.overlap_status.value, _json(ledger.supporting_evidence), ledger.unresolved_uncertainty, ledger.safe_wording, _json(ledger.prohibited_wording), ledger.review_status, _json_provenance(ledger.provenance)),
        )

    def novelty_claim_ledgers(self, run_id: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM novelty_claim_ledgers WHERE run_id=? ORDER BY candidate_id", (run_id,))

    # Experiment methods
    def save_experiment_spec(self, spec: ExperimentSpec) -> None:
        self.execute(
            "INSERT OR REPLACE INTO experiment_specs(id,project_id,run_id,candidate_id,hypothesis,causal_mechanism,falsification_condition,repository,baseline_commit,dataset_identity,dataset_checksum,split_definition,baseline_configuration,proposed_change,independent_variables_json,controlled_variables_json,metrics_json,random_seeds_json,ablations_json,robustness_tests_json,statistical_plan,compute_budget,stopping_conditions_json,expected_outputs_json,artifact_contract,provenance_json,metric_direction_json,minimum_meaningful_effect_json,required_seed_count,network_policy,target_worker,approval_required,gap_id,methodology_freeze_id,interpretation_rules_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (spec.id, spec.project_id, spec.run_id, spec.candidate_id, spec.hypothesis, spec.causal_mechanism, spec.falsification_condition, spec.repository, spec.baseline_commit, spec.dataset_identity, spec.dataset_checksum, spec.split_definition, spec.baseline_configuration, spec.proposed_change, _json(spec.independent_variables), _json(spec.controlled_variables), _json(spec.metrics), _json(spec.random_seeds), _json(spec.ablations), _json(spec.robustness_tests), spec.statistical_plan, spec.compute_budget, _json(spec.stopping_conditions), _json(spec.expected_outputs), spec.artifact_contract, _json_provenance(spec.provenance), _json(spec.metric_direction), _json(spec.minimum_meaningful_effect), spec.required_seed_count, spec.network_policy, spec.target_worker, int(spec.approval_required), spec.gap_id, spec.methodology_freeze_id, _json(spec.interpretation_rules))
        )

    def get_experiment_spec(self, candidate_id: str) -> ExperimentSpec | None:
        row = self.row("SELECT * FROM experiment_specs WHERE candidate_id=?", (candidate_id,))
        return _experiment_spec(row) if row else None

    def get_experiment_spec_by_id(self, spec_id: str) -> ExperimentSpec | None:
        row = self.row("SELECT * FROM experiment_specs WHERE id=?", (spec_id,))
        return _experiment_spec(row) if row else None

    def save_environment_manifest(self, manifest: EnvironmentManifest) -> None:
        self.execute(
            "INSERT INTO environment_manifests(id,project_id,run_id,spec_id,operating_system,python_version,environment_manager,dependency_lock_checksum,cuda_compatibility,gpu_model,gpu_memory,disk_space,dataset_paths_json,sample_counts_json,splits_json,pretrained_checkpoint_identity,reproducibility_flags,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (manifest.id, manifest.project_id, manifest.run_id, manifest.spec_id, manifest.operating_system, manifest.python_version, manifest.environment_manager, manifest.dependency_lock_checksum, manifest.cuda_compatibility, manifest.gpu_model, manifest.gpu_memory, manifest.disk_space, _json(manifest.dataset_paths), _json(manifest.sample_counts), _json(manifest.splits), manifest.pretrained_checkpoint_identity, manifest.reproducibility_flags, _json_provenance(manifest.provenance))
        )

    def get_environment_manifest(self, spec_id: str) -> EnvironmentManifest | None:
        row = self.row("SELECT * FROM environment_manifests WHERE spec_id=?", (spec_id,))
        return _environment_manifest(row) if row else None

    def save_validation_report(self, report: ValidationReport) -> None:
        self.execute(
            "INSERT INTO validation_reports(id,project_id,run_id,manifest_id,schema_validation_passed,static_checks_passed,unit_tests_passed,import_test_passed,one_batch_forward_passed,backward_gradient_passed,checkpoint_save_reload_passed,short_smoke_training_passed,evaluation_parsing_passed,budget_approval_passed,details_json,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (report.id, report.project_id, report.run_id, report.manifest_id, int(report.schema_validation_passed), int(report.static_checks_passed), int(report.unit_tests_passed), int(report.import_test_passed), int(report.one_batch_forward_passed), int(report.backward_gradient_passed), int(report.checkpoint_save_reload_passed), int(report.short_smoke_training_passed), int(report.evaluation_parsing_passed), int(report.budget_approval_passed), _json(report.details), _json_provenance(report.provenance))
        )

    def get_validation_report(self, manifest_id: str) -> ValidationReport | None:
        row = self.row("SELECT * FROM validation_reports WHERE manifest_id=?", (manifest_id,))
        return _validation_report(row) if row else None

    def save_execution_job(self, job: ExecutionJob) -> None:
        self.execute(
            "INSERT INTO execution_jobs(id,project_id,run_id,spec_id,worker,stage,status,started_at,completed_at,timeout_seconds,retry_policy) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,completed_at=excluded.completed_at",
            (job.id, job.project_id, job.run_id, job.spec_id, job.worker, job.stage, job.status.value, job.started_at, job.completed_at, job.timeout_seconds, job.retry_policy)
        )

    def get_execution_job(self, job_id: str) -> ExecutionJob | None:
        row = self.row("SELECT * FROM execution_jobs WHERE id=?", (job_id,))
        return _execution_job(row) if row else None

    def execution_jobs(self, spec_id: str) -> list[ExecutionJob]:
        return [_execution_job(row) for row in self.rows("SELECT * FROM execution_jobs WHERE spec_id=? ORDER BY started_at", (spec_id,))]

    def save_execution_result(self, result: ExecutionResult) -> None:
        self.execute(
            "INSERT INTO execution_results(id,project_id,run_id,job_id,exit_code,stdout,stderr,structured_metrics_json,gpu_telemetry_json,checkpoint_manifest_json,failure_classification,command_json,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result.id, result.project_id, result.run_id, result.job_id, result.exit_code, result.stdout, result.stderr, _json(result.structured_metrics), _json(result.gpu_telemetry), _json(result.checkpoint_manifest), result.failure_classification, result.command_json, _json_provenance(result.provenance))
        )

    def get_execution_result(self, job_id: str) -> ExecutionResult | None:
        row = self.row("SELECT * FROM execution_results WHERE job_id=?", (job_id,))
        return _execution_result(row) if row else None

    def execution_commands_for_spec(self, spec_id: str) -> list[str]:
        """Exact executed commands for a spec, from the executor's recorded
        argv. Empty when nothing was executed (e.g. legacy rows); callers must
        report absence honestly, never substitute a reconstructed command."""
        commands: list[str] = []
        for job in self.execution_jobs(spec_id):
            result = self.get_execution_result(job.id)
            if result is None:
                continue
            argv = _loads(result.command_json, None)
            if isinstance(argv, list) and argv:
                commands.append(" ".join(str(part) for part in argv))
        return commands

    def save_statistical_analysis(self, analysis: StatisticalAnalysis) -> None:
        self.execute(
            "INSERT INTO statistical_analyses(id,project_id,run_id,spec_id,baseline_metrics_json,proposed_metrics_json,effect_size_json,confidence_intervals_json,seed_level_results_json,ablation_contributions_json,parameter_count_diff,flops_latency_diff,training_inference_cost,failure_case_analysis,hypothesis_support_assessment,provenance_json,decision_gates_json,statistical_tests_json,evidence_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (analysis.id, analysis.project_id, analysis.run_id, analysis.spec_id, _json(analysis.baseline_metrics), _json(analysis.proposed_metrics), _json(analysis.effect_size), _json(analysis.confidence_intervals), _json(analysis.seed_level_results), _json(analysis.ablation_contributions), analysis.parameter_count_diff, analysis.flops_latency_diff, analysis.training_inference_cost, analysis.failure_case_analysis, analysis.hypothesis_support_assessment, _json_provenance(analysis.provenance), _json(dict(analysis.decision_gates)), _json(dict(analysis.statistical_tests)), analysis.evidence_status)
        )

    def get_statistical_analysis(self, spec_id: str) -> StatisticalAnalysis | None:
        row = self.row("SELECT * FROM statistical_analyses WHERE spec_id=?", (spec_id,))
        return _statistical_analysis(row) if row else None

    def save_experiment_report(self, report: ExperimentReport) -> None:
        self.execute(
            "INSERT INTO experiment_reports(id,project_id,run_id,analysis_id,literature_motivation,research_gap,hypothesis,proposed_method,implementation_diff,lineage,dataset_split,exact_commands_json,baseline_results_json,aggregated_results_json,uncertainty,ablations,robustness_tests,failed_runs,cost,limitations,reproducibility_instructions,decision,decision_evidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (report.id, report.project_id, report.run_id, report.analysis_id, report.literature_motivation, report.research_gap, report.hypothesis, report.proposed_method, report.implementation_diff, report.lineage, report.dataset_split, _json(report.exact_commands), _json(report.baseline_results), _json(report.aggregated_results), report.uncertainty, report.ablations, report.robustness_tests, report.failed_runs, report.cost, _json(report.limitations), report.reproducibility_instructions, report.decision.value, report.decision_evidence, _json_provenance(report.provenance))
        )

    def get_experiment_report(self, analysis_id: str) -> ExperimentReport | None:
        row = self.row("SELECT * FROM experiment_reports WHERE analysis_id=?", (analysis_id,))
        return _experiment_report(row) if row else None

    def save_manuscript(self, manuscript: ManuscriptRecord) -> None:
        self.execute(
            "INSERT INTO manuscripts(id,project_id,run_id,title,status,decision,lineage_json,evidence_bundle_path,markdown_path,latex_path,bibtex_path,manifest_path,pdf_path,gaps_json,provenance_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (manuscript.id, manuscript.project_id, manuscript.run_id, manuscript.title, manuscript.status, manuscript.decision, manuscript.lineage_json, manuscript.evidence_bundle_path, manuscript.markdown_path, manuscript.latex_path, manuscript.bibtex_path, manuscript.manifest_path, manuscript.pdf_path, manuscript.gaps_json, _json_provenance(manuscript.provenance), manuscript.created_at)
        )

    def latest_manuscript(self, project_id: str) -> ManuscriptRecord | None:
        row = self.row("SELECT * FROM manuscripts WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project_id,))
        if not row:
            return None
        return ManuscriptRecord(
            id=row["id"], project_id=row["project_id"], run_id=row["run_id"], title=row["title"],
            status=row["status"], decision=row["decision"], lineage_json=row["lineage_json"],
            evidence_bundle_path=row["evidence_bundle_path"], markdown_path=row["markdown_path"],
            latex_path=row["latex_path"], bibtex_path=row["bibtex_path"], manifest_path=row["manifest_path"],
            pdf_path=row["pdf_path"], gaps_json=row["gaps_json"],
            provenance=ArtifactProvenance(**json.loads(row["provenance_json"])), created_at=row["created_at"],
        )

    def save_finding(self, finding: ResearchFinding) -> None:
        self.execute(
            "INSERT INTO research_findings(id,project_id,run_id,status,question_id,candidate_id,methodology_freeze_id,"
            "spec_id,analysis_id,report_id,hypothesis,expected_observation,implementation_digest,observed_outcome_json,"
            "interpretation,uncertainty,fidelity,failure_classification,lesson,applicability,open_questions_json,"
            "next_action,keywords_json,provenance_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (finding.id, finding.project_id, finding.run_id, finding.status.value, finding.question_id, finding.candidate_id,
             finding.methodology_freeze_id, finding.spec_id, finding.analysis_id, finding.report_id, finding.hypothesis,
             finding.expected_observation, finding.implementation_digest, finding.observed_outcome_json, finding.interpretation,
             finding.uncertainty, finding.fidelity, finding.failure_classification, finding.lesson, finding.applicability,
             finding.open_questions_json, finding.next_action, finding.keywords_json, _json_provenance(finding.provenance),
             finding.created_at),
        )

    def findings_for_project(self, project_id: str, limit: int = 100) -> list[ResearchFinding]:
        rows = self.rows(
            "SELECT * FROM research_findings WHERE project_id=? ORDER BY rowid DESC LIMIT ?", (project_id, limit)
        )
        return [_finding(row) for row in rows]

    def findings_for_question(self, project_id: str, question_id: str, limit: int = 8) -> list[ResearchFinding]:
        rows = self.rows(
            "SELECT * FROM research_findings WHERE project_id=? AND question_id=? "
            "ORDER BY CASE status WHEN 'refuted_hypothesis' THEN 0 WHEN 'implementation_failure' THEN 1 "
            "WHEN 'inconclusive' THEN 2 ELSE 3 END, rowid DESC LIMIT ?",
            (project_id, question_id, limit),
        )
        return [_finding(row) for row in rows]

    def search_findings(self, project_id: str, terms: Sequence[str], limit: int = 8) -> list[ResearchFinding]:
        """Lexical retrieval over the Findings Memory (simple + inspectable).

        Matches any term against hypothesis/lesson/method keywords; failures
        and refutations rank first so the generator sees what NOT to redo.
        """
        if not terms:
            return []
        clauses = " OR ".join(
            "(hypothesis LIKE ? OR lesson LIKE ? OR keywords_json LIKE ?)" for _ in terms
        )
        params: list[str] = []
        for term in terms:
            like = f"%{term}%"
            params.extend([like, like, like])
        rows = self.rows(
            "SELECT * FROM research_findings WHERE project_id=? AND (" + clauses + ") "
            "ORDER BY CASE status WHEN 'refuted_hypothesis' THEN 0 WHEN 'implementation_failure' THEN 1 "
            "WHEN 'inconclusive' THEN 2 ELSE 3 END, rowid DESC LIMIT ?",
            (project_id, *params, limit),
        )
        return [_finding(row) for row in rows]

    def save_reinvention_analysis(self, analysis: ReinventionAnalysis) -> None:
        self.execute(
            "INSERT INTO reinvention_analyses(id,project_id,run_id,spec_id,report_id,parent_candidate_id,action,root_cause_diagnosis,empirical_lessons_json,next_generation_hypothesis,revised_mechanism,actionable_next_steps_json,next_candidate_id,provenance_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (analysis.id, analysis.project_id, analysis.run_id, analysis.spec_id, analysis.report_id, analysis.parent_candidate_id, analysis.action.value, analysis.root_cause_diagnosis, _json(analysis.empirical_lessons), analysis.next_generation_hypothesis, analysis.revised_mechanism, _json(analysis.actionable_next_steps), analysis.next_candidate_id, _json_provenance(analysis.provenance), utc_now())
        )

    def get_reinvention_analysis(self, report_id: str) -> ReinventionAnalysis | None:
        row = self.row("SELECT * FROM reinvention_analyses WHERE report_id=? ORDER BY created_at DESC LIMIT 1", (report_id,))
        return _reinvention_analysis(row) if row else None

    def get_reinvention_analysis_by_id(self, reinvention_id: str) -> ReinventionAnalysis | None:
        row = self.row("SELECT * FROM reinvention_analyses WHERE id=?", (reinvention_id,))
        return _reinvention_analysis(row) if row else None

    def get_reinvention_analysis_by_candidate(self, candidate_id: str) -> ReinventionAnalysis | None:
        row = self.row("SELECT * FROM reinvention_analyses WHERE parent_candidate_id=? ORDER BY created_at DESC LIMIT 1", (candidate_id,))
        return _reinvention_analysis(row) if row else None

    def save_worker_job(self, job: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO worker_jobs(id,project_id,run_id,spec_id,worker,lease_id,state,attempt,checksum,actions_json,timeout_seconds,created_at,updated_at,heartbeat_at,lease_expires_at,process_id,process_group_id,last_log_offset,error_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,attempt=excluded.attempt,updated_at=excluded.updated_at,heartbeat_at=excluded.heartbeat_at,lease_expires_at=excluded.lease_expires_at,process_id=excluded.process_id,process_group_id=excluded.process_group_id,last_log_offset=excluded.last_log_offset,error_json=excluded.error_json",
            (job["id"], job["project_id"], job["run_id"], job["spec_id"], job["worker"], job.get("lease_id"), job["state"], job.get("attempt", 0), job["checksum"], _json(job.get("actions", [])), job.get("timeout_seconds", 3600), job.get("created_at", utc_now()), job.get("updated_at", utc_now()), job.get("heartbeat_at"), job.get("lease_expires_at"), job.get("process_id"), job.get("process_group_id"), job.get("last_log_offset", 0), _json(job.get("error")) if job.get("error") else None),
        )

    def worker_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.row("SELECT * FROM worker_jobs WHERE id=?", (job_id,))
        if row:
            row["actions"] = _loads(row.pop("actions_json"), [])
            row["error"] = _loads(row.pop("error_json"), None)
        return row

    def worker_jobs_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [self.worker_job(row["id"]) for row in self.rows("SELECT id FROM worker_jobs WHERE run_id=? ORDER BY created_at", (run_id,)) if self.worker_job(row["id"])]

    def append_worker_transition(self, run_id: str, from_state: str | None, to_state: str, *, reason: str = "", worker_id: str | None = None, attempt: int = 0, event_id: str | None = None) -> int:
        with self._lock, self.connect() as connection:
            cursor = connection.execute("INSERT INTO experiment_state_transitions(run_id,from_state,to_state,reason,created_at,worker_id,attempt,event_id) VALUES(?,?,?,?,?,?,?,?)", (run_id, from_state, to_state, reason, utc_now(), worker_id, attempt, event_id))
            return int(cursor.lastrowid)

    def worker_transitions(self, run_id: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM experiment_state_transitions WHERE run_id=? ORDER BY id", (run_id,))

    def save_exact_approval(self, approval: dict[str, Any]) -> dict[str, Any]:
        existing = self.row("SELECT * FROM experiment_approvals WHERE run_id=? AND specification_checksum=?", (approval["run_id"], approval["specification_checksum"]))
        if existing:
            return existing
        self.execute("INSERT INTO experiment_approvals(id,run_id,specification_checksum,approver,approved_scope,approved_target,approved_budget,expires_at,created_at,status,revoked_at,approved_actions_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (approval["id"], approval["run_id"], approval["specification_checksum"], approval["approver"], approval["approved_scope"], approval["approved_target"], approval["approved_budget"], approval["expires_at"], approval.get("created_at", utc_now()), approval.get("status", "pending"), approval.get("revoked_at"), _json(approval.get("approved_actions", []))))
        return self.row("SELECT * FROM experiment_approvals WHERE id=?", (approval["id"],)) or approval

    def exact_approval(self, run_id: str, checksum: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM experiment_approvals WHERE run_id=?"
        params: tuple[Any, ...] = (run_id,)
        if checksum:
            query += " AND specification_checksum=?"
            params += (checksum,)
        row = self.row(query + " ORDER BY created_at DESC LIMIT 1", params)
        if row:
            row["approved_actions"] = _loads(row.pop("approved_actions_json"), [])
        return row

    def save_methodology_freeze(self, freeze: "MethodologyFreeze") -> None:
        from autoresearch.domain.models import MethodologyFreeze  # local import keeps module load order simple
        assert isinstance(freeze, MethodologyFreeze)
        self.execute(
            "INSERT INTO methodology_freezes(id,project_id,run_id,candidate_id,version,selection_rationale,prompt_version,artifact_path,checksum,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (freeze.id, freeze.project_id, freeze.run_id, freeze.candidate_id, freeze.version, freeze.selection_rationale, freeze.prompt_version, freeze.artifact_path, freeze.checksum, freeze.created_at),
        )

    def methodology_freeze(self, freeze_id: str) -> "MethodologyFreeze | None":
        row = self.row("SELECT * FROM methodology_freezes WHERE id=?", (freeze_id,))
        return _methodology_freeze(row) if row else None

    def latest_methodology_freeze(self, project_id: str) -> "MethodologyFreeze | None":
        # rowid ordering = insertion order; created_at alone could tie within
        # timestamp resolution across different candidates.
        row = self.row("SELECT * FROM methodology_freezes WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project_id,))
        return _methodology_freeze(row) if row else None

    def latest_methodology_freeze_for_candidate(self, candidate_id: str) -> "MethodologyFreeze | None":
        row = self.row("SELECT * FROM methodology_freezes WHERE candidate_id=? ORDER BY version DESC LIMIT 1", (candidate_id,))
        return _methodology_freeze(row) if row else None

    def approval_by_checksum(self, checksum: str) -> dict[str, Any] | None:
        # Not run-scoped by design: the specification checksum embeds the full
        # spec (including project_id, run_id, and a fresh spec id), actions, and
        # worker identity, so a checksum can only match the exact spec it was
        # approved for. Do not weaken the checksum construction.
        row = self.row("SELECT * FROM experiment_approvals WHERE specification_checksum=? ORDER BY created_at DESC LIMIT 1", (checksum,))
        if row:
            row["approved_actions"] = _loads(row.pop("approved_actions_json"), [])
        return row

    def approve_exact(self, approval_id: str, *, approver: str) -> dict[str, Any] | None:
        self.execute("UPDATE experiment_approvals SET status='approved',approver=? WHERE id=? AND status='pending'", (approver, approval_id))
        row = self.row("SELECT * FROM experiment_approvals WHERE id=?", (approval_id,))
        if row:
            row["approved_actions"] = _loads(row.pop("approved_actions_json"), [])
        return row


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _json_provenance(value: ArtifactProvenance) -> str:
    return _json({"project_id": value.project_id, "run_id": value.run_id, "source_evidence": value.source_evidence, "provider": value.provider, "model": value.model, "prompt_version": value.prompt_version, "created_at": value.created_at, "checksum": value.checksum})


def _provenance(value: str) -> ArtifactProvenance:
    data = _loads(value, {})
    data["source_evidence"] = tuple(data.get("source_evidence") or [])
    return ArtifactProvenance(**data)


def _paper(row: dict[str, Any]) -> PaperRecord:
    return PaperRecord(row["id"], row["project_id"], row["run_id"], row["title"], row["abstract"], tuple(_loads(row["authors_json"], [])), row["year"], row["publication_date"], row["doi"], row["arxiv_id"], row["url"], tuple(_loads(row["source_names_json"], [])), row["open_access_pdf_url"], bool(row["is_open_access"]), int(row["citation_count"]), row["venue"], row["checksum"], row["created_at"])


def _download(row: dict[str, Any]) -> DownloadRecord:
    return DownloadRecord(row["id"], row["project_id"], row["run_id"], row["paper_id"], DownloadStatus(row["status"]), row["source"], row["url"], row["legal_basis"], row["license"], row["path"], row["reason"], row["checksum"], row["created_at"])


def _review(row: dict[str, Any]) -> PaperReview:
    return PaperReview(row["id"], row["project_id"], row["run_id"], row["paper_id"], row["problem"], row["research_question"], row["method"], tuple(_loads(row["datasets_json"], [])), row["experimental_protocol"], tuple(_loads(row["results_json"], [])), tuple(_loads(row["ablations_json"], [])), tuple(_loads(row["limitations_json"], [])), tuple(_loads(row["failure_modes_json"], [])), tuple(_loads(row["evidence_claim_ids_json"], [])), float(row["confidence"]), _provenance(row["provenance_json"]))


def _claim(row: dict[str, Any]) -> EvidenceClaim:
    return EvidenceClaim(row["id"], row["project_id"], row["run_id"], row["paper_id"], row["claim_type"], row["statement"], row["excerpt"], row["page"], row["section"], float(row["confidence"]), _provenance(row["provenance_json"]))


def _score(row: dict[str, Any]) -> PriorityScore:
    return PriorityScore(row["id"], row["project_id"], row["run_id"], row["paper_id"], float(row["semantic_similarity"]), float(row["methodological_similarity"]), float(row["research_question_similarity"]), float(row["recency"]), float(row["influence"]), float(row["evidence_quality"]), float(row["total"]), _loads(row["explanation_json"], {}), _provenance(row["provenance_json"]))


def _gap(row: dict[str, Any]) -> ResearchGap:
    return ResearchGap(row["id"], row["project_id"], row["run_id"], row["statement"], tuple(_loads(row["supporting_paper_ids_json"], [])), tuple(_loads(row["supporting_claim_ids_json"], [])), row["why_unsolved"], row["missing_experiment"], row["falsification_test"], float(row["confidence"]), tuple(_loads(row["counter_evidence_claim_ids_json"], [])), GapClassification(row["classification"]), _provenance(row["provenance_json"]))


def _novelty_candidate(row: dict[str, Any]) -> NoveltyCandidate:
    return NoveltyCandidate(row["id"], row["project_id"], row["run_id"], row["gap_id"], row["strategy"], row["method"], row["causal_mechanism"], row["target_limitation"], row["difference_from_existing"], row["expected_effect"], row["falsification_condition"], row["minimum_validating_experiment"], tuple(_loads(row["required_ablations_json"], [])), row["implementation_complexity"], row["expected_compute"], tuple(_loads(row["risks_and_objections_json"], [])), CandidateStatus(row["status"]), _provenance(row["provenance_json"]))

def _prior_art_assessment(row: dict[str, Any]) -> PriorArtAssessment:
    return PriorArtAssessment(row["id"], row["project_id"], row["run_id"], row["candidate_id"], tuple(_loads(row["closest_paper_ids_json"], [])), PriorArtStatus(row["overlap_status"]), row["mechanism_comparison"], tuple(_loads(row["supporting_excerpts_json"], [])), row["unresolved_overlap"], _provenance(row["provenance_json"]))

def _novelty_critique(row: dict[str, Any]) -> NoveltyCritique:
    return NoveltyCritique(row["id"], row["project_id"], row["run_id"], row["candidate_id"], float(row["novelty_score"]), float(row["significance_score"]), float(row["causal_plausibility_score"]), float(row["falsifiability_score"]), float(row["experimental_discriminability_score"]), float(row["feasibility_score"]), float(row["implementation_risk_score"]), float(row["expected_information_gain_score"]), float(row["compute_efficiency_score"]), float(row["publication_potential_score"]), tuple(_loads(row["objections_json"], [])), tuple(_loads(row["failure_scenarios_json"], [])), _provenance(row["provenance_json"]))

def _candidate_revision(row: dict[str, Any]) -> CandidateRevision:
    return CandidateRevision(row["id"], row["project_id"], row["run_id"], row["original_candidate_id"], row["revised_candidate_id"], row["critique_id"], int(row["revision_count"]), row["changes_made"], _provenance(row["provenance_json"]))

def _novelty_portfolio(row: dict[str, Any]) -> NoveltyPortfolio:
    return NoveltyPortfolio(row["id"], row["project_id"], row["run_id"], row["low_risk_candidate_id"], row["balanced_candidate_id"], row["high_risk_candidate_id"], _loads(row["selection_reasons_json"], {}), _provenance(row["provenance_json"]))

def _methodology_freeze(row: dict[str, Any]) -> "MethodologyFreeze":
    from autoresearch.domain.models import MethodologyFreeze
    return MethodologyFreeze(row["id"], row["project_id"], row["run_id"], row["candidate_id"], int(row["version"]), row["selection_rationale"], row["prompt_version"], row["artifact_path"], row["checksum"], row["created_at"])


def _experiment_spec(row: dict[str, Any]) -> ExperimentSpec:
    return ExperimentSpec(row["id"], row["project_id"], row["run_id"], row["candidate_id"], row["hypothesis"], row["causal_mechanism"], row["falsification_condition"], row["repository"], row["baseline_commit"], row["dataset_identity"], row["dataset_checksum"], row["split_definition"], row["baseline_configuration"], row["proposed_change"], tuple(_loads(row["independent_variables_json"], [])), tuple(_loads(row["controlled_variables_json"], [])), tuple(_loads(row["metrics_json"], [])), tuple(_loads(row["random_seeds_json"], [])), tuple(_loads(row["ablations_json"], [])), tuple(_loads(row["robustness_tests_json"], [])), row["statistical_plan"], row["compute_budget"], tuple(_loads(row["stopping_conditions_json"], [])), tuple(_loads(row["expected_outputs_json"], [])), row["artifact_contract"], _provenance(row["provenance_json"]), _loads(row.get("metric_direction_json"), {}), _loads(row.get("minimum_meaningful_effect_json"), {}), int(row.get("required_seed_count", 1)), row.get("network_policy", "disabled"), row.get("target_worker", "local"), bool(row.get("approval_required", 1)), str(row.get("gap_id") or ""), str(row.get("methodology_freeze_id") or ""), tuple(_loads(row.get("interpretation_rules_json"), [])))

def _environment_manifest(row: dict[str, Any]) -> EnvironmentManifest:
    return EnvironmentManifest(row["id"], row["project_id"], row["run_id"], row["spec_id"], row["operating_system"], row["python_version"], row["environment_manager"], row["dependency_lock_checksum"], row["cuda_compatibility"], row["gpu_model"], row["gpu_memory"], row["disk_space"], tuple(_loads(row["dataset_paths_json"], [])), _loads(row["sample_counts_json"], {}), tuple(_loads(row["splits_json"], [])), row["pretrained_checkpoint_identity"], row["reproducibility_flags"], _provenance(row["provenance_json"]))

def _validation_report(row: dict[str, Any]) -> ValidationReport:
    return ValidationReport(row["id"], row["project_id"], row["run_id"], row["manifest_id"], bool(row["schema_validation_passed"]), bool(row["static_checks_passed"]), bool(row["unit_tests_passed"]), bool(row["import_test_passed"]), bool(row["one_batch_forward_passed"]), bool(row["backward_gradient_passed"]), bool(row["checkpoint_save_reload_passed"]), bool(row["short_smoke_training_passed"]), bool(row["evaluation_parsing_passed"]), bool(row["budget_approval_passed"]), _loads(row["details_json"], {}), _provenance(row["provenance_json"]))

def _execution_job(row: dict[str, Any]) -> ExecutionJob:
    return ExecutionJob(row["id"], row["project_id"], row["run_id"], row["spec_id"], row["worker"], row["stage"], RunStatus(row["status"]), row["started_at"], row["completed_at"], int(row["timeout_seconds"]), row["retry_policy"])

def _execution_result(row: dict[str, Any]) -> ExecutionResult:
    return ExecutionResult(row["id"], row["project_id"], row["run_id"], row["job_id"], int(row["exit_code"]), row["stdout"], row["stderr"], _loads(row["structured_metrics_json"], {}), _loads(row["gpu_telemetry_json"], {}), tuple(_loads(row["checkpoint_manifest_json"], [])), row["failure_classification"], _provenance(row["provenance_json"]), command_json=str(row.get("command_json") or ""))

def _finding(row: dict[str, Any]) -> ResearchFinding:
    return ResearchFinding(
        id=row["id"], project_id=row["project_id"], run_id=row["run_id"],
        status=FindingStatus(str(row["status"])), question_id=row["question_id"],
        candidate_id=row["candidate_id"], methodology_freeze_id=row["methodology_freeze_id"],
        spec_id=row["spec_id"], analysis_id=row["analysis_id"], report_id=row["report_id"],
        hypothesis=row["hypothesis"], expected_observation=row["expected_observation"],
        implementation_digest=row["implementation_digest"], observed_outcome_json=row["observed_outcome_json"],
        interpretation=row["interpretation"], uncertainty=row["uncertainty"], fidelity=row["fidelity"],
        failure_classification=row["failure_classification"], lesson=row["lesson"],
        applicability=row["applicability"], open_questions_json=row["open_questions_json"],
        next_action=row["next_action"], keywords_json=row["keywords_json"],
        provenance=ArtifactProvenance(**json.loads(row["provenance_json"])), created_at=row["created_at"],
    )


def _statistical_analysis(row: dict[str, Any]) -> StatisticalAnalysis:
    assessment = str(row["hypothesis_support_assessment"] or "")
    return StatisticalAnalysis(
        row["id"], row["project_id"], row["run_id"], row["spec_id"],
        _loads(row["baseline_metrics_json"], {}), _loads(row["proposed_metrics_json"], {}),
        _loads(row["effect_size_json"], {}), _loads(row["confidence_intervals_json"], {}),
        tuple(_loads(row["seed_level_results_json"], [])), _loads(row["ablation_contributions_json"], {}),
        int(row["parameter_count_diff"]),
        float(row["flops_latency_diff"]) if row["flops_latency_diff"] is not None else None,
        float(row["training_inference_cost"]), row["failure_case_analysis"], assessment,
        _provenance(row["provenance_json"]),
        decision_gates=_loads(row.get("decision_gates_json") or "{}", {}),
        statistical_tests=_loads(row.get("statistical_tests_json") or "{}", {}),
        # Pre-V13 rows never persisted the status: the assessment is the
        # honest fallback rather than defaulting to insufficiency.
        evidence_status=str(row.get("evidence_status") or assessment or "INSUFFICIENT_EVIDENCE"),
    )

def _report_limitations(value: Any) -> tuple[str, ...]:
    """Limitations column: JSON list for new reports, legacy plain text for
    pre-v15 rows. Never returns an invented entry."""
    parsed = _loads(value, None)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed if str(item).strip())
    return (str(value),) if value else ()


def _experiment_report(row: dict[str, Any]) -> ExperimentReport:
    return ExperimentReport(row["id"], row["project_id"], row["run_id"], row["analysis_id"], row["literature_motivation"], row["research_gap"], row["hypothesis"], row["proposed_method"], row["implementation_diff"], row["lineage"], row["dataset_split"], tuple(_loads(row["exact_commands_json"], [])), _loads(row["baseline_results_json"], {}), _loads(row["aggregated_results_json"], {}), row["uncertainty"], row["ablations"], row["robustness_tests"], int(row["failed_runs"]), float(row["cost"]), _report_limitations(row["limitations"]), row["reproducibility_instructions"], ExperimentDecision(row["decision"]), row["decision_evidence"], _provenance(row["provenance_json"]))


def _reinvention_analysis(row: dict[str, Any]) -> ReinventionAnalysis:
    return ReinventionAnalysis(
        row["id"], row["project_id"], row["run_id"], row["spec_id"],
        row["report_id"], row["parent_candidate_id"], ReinventionAction(row["action"]),
        row["root_cause_diagnosis"], tuple(_loads(row["empirical_lessons_json"], [])),
        row["next_generation_hypothesis"], row["revised_mechanism"],
        tuple(_loads(row["actionable_next_steps_json"], [])),
        _provenance(row["provenance_json"]), row.get("next_candidate_id"),
    )


_SCHEMA_V1 = """
CREATE TABLE projects(id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE pipeline_runs(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),kind TEXT NOT NULL,status TEXT NOT NULL,input_json TEXT NOT NULL,started_at TEXT NOT NULL,completed_at TEXT,error TEXT);
CREATE INDEX pipeline_runs_project_idx ON pipeline_runs(project_id,started_at);
CREATE TABLE seed_papers(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),input_type TEXT NOT NULL,input_value TEXT NOT NULL,title TEXT NOT NULL,text TEXT NOT NULL,topic TEXT NOT NULL,research_question TEXT NOT NULL,doi TEXT,url TEXT,methods_json TEXT NOT NULL,datasets_json TEXT NOT NULL,citations_json TEXT NOT NULL,limitations_json TEXT NOT NULL,checksum TEXT NOT NULL,provenance_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE search_events(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),source TEXT NOT NULL,strategy TEXT NOT NULL,query TEXT NOT NULL,requested_limit INTEGER NOT NULL,result_count INTEGER NOT NULL,raw_checksum TEXT NOT NULL,error TEXT,retrieved_at TEXT NOT NULL);
CREATE TABLE papers(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),title TEXT NOT NULL,abstract TEXT NOT NULL,authors_json TEXT NOT NULL,year INTEGER,publication_date TEXT,doi TEXT,arxiv_id TEXT,url TEXT,source_names_json TEXT NOT NULL,open_access_pdf_url TEXT,is_open_access INTEGER NOT NULL,citation_count INTEGER NOT NULL,venue TEXT NOT NULL,checksum TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX papers_project_date_idx ON papers(project_id,publication_date);
CREATE INDEX papers_doi_idx ON papers(doi);
CREATE INDEX papers_arxiv_idx ON papers(arxiv_id);
CREATE TABLE search_candidates(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),source TEXT NOT NULL,strategy TEXT NOT NULL,source_rank INTEGER NOT NULL,query TEXT NOT NULL,raw_checksum TEXT NOT NULL,retrieved_at TEXT NOT NULL,UNIQUE(run_id,paper_id,source,strategy));
CREATE TABLE discovery_scores(project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),components_json TEXT NOT NULL,total REAL NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(run_id,paper_id));
CREATE TABLE download_records(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),status TEXT NOT NULL,source TEXT NOT NULL,url TEXT,legal_basis TEXT,license TEXT,path TEXT,reason TEXT,checksum TEXT,created_at TEXT NOT NULL);
CREATE TABLE evidence_claims(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),claim_type TEXT NOT NULL,statement TEXT NOT NULL,excerpt TEXT NOT NULL,page INTEGER,section TEXT,confidence REAL NOT NULL,provenance_json TEXT NOT NULL);
CREATE TABLE paper_reviews(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),problem TEXT NOT NULL,research_question TEXT NOT NULL,method TEXT NOT NULL,datasets_json TEXT NOT NULL,experimental_protocol TEXT NOT NULL,results_json TEXT NOT NULL,ablations_json TEXT NOT NULL,limitations_json TEXT NOT NULL,failure_modes_json TEXT NOT NULL,evidence_claim_ids_json TEXT NOT NULL,confidence REAL NOT NULL,provenance_json TEXT NOT NULL,UNIQUE(run_id,paper_id));
CREATE TABLE review_failures(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),reason TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE priority_scores(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),paper_id TEXT NOT NULL REFERENCES papers(id),semantic_similarity REAL NOT NULL,methodological_similarity REAL NOT NULL,research_question_similarity REAL NOT NULL,recency REAL NOT NULL,influence REAL NOT NULL,evidence_quality REAL NOT NULL,total REAL NOT NULL,explanation_json TEXT NOT NULL,provenance_json TEXT NOT NULL,UNIQUE(run_id,paper_id));
CREATE TABLE research_gaps(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),statement TEXT NOT NULL,supporting_paper_ids_json TEXT NOT NULL,supporting_claim_ids_json TEXT NOT NULL,why_unsolved TEXT NOT NULL,missing_experiment TEXT NOT NULL,falsification_test TEXT NOT NULL,confidence REAL NOT NULL,counter_evidence_claim_ids_json TEXT NOT NULL,classification TEXT NOT NULL,provenance_json TEXT NOT NULL);
CREATE TABLE artifacts(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),run_id TEXT NOT NULL REFERENCES pipeline_runs(id),artifact_type TEXT NOT NULL,path TEXT NOT NULL,source_evidence_json TEXT NOT NULL,provider TEXT NOT NULL,model TEXT NOT NULL,prompt_version TEXT NOT NULL,checksum TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE provider_usage(id TEXT PRIMARY KEY,project_id TEXT,run_id TEXT,operation TEXT NOT NULL,role TEXT NOT NULL,provider TEXT NOT NULL,model TEXT NOT NULL,prompt_tokens INTEGER,completion_tokens INTEGER,estimated_cost REAL,currency TEXT,request_id TEXT,latency_ms REAL,created_at TEXT NOT NULL);
CREATE TABLE project_budgets(project_id TEXT PRIMARY KEY REFERENCES projects(id),hard_stop REAL,currency TEXT NOT NULL,updated_at TEXT NOT NULL);
"""

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS pipeline_checkpoints(
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  stage TEXT NOT NULL,
  item_key TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(run_id,stage,item_key)
);
CREATE INDEX IF NOT EXISTS pipeline_checkpoints_status_idx ON pipeline_checkpoints(run_id,status,stage);
CREATE INDEX IF NOT EXISTS download_records_project_paper_idx ON download_records(project_id,paper_id,created_at);
CREATE INDEX IF NOT EXISTS paper_reviews_project_paper_idx ON paper_reviews(project_id,paper_id);
"""

_SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS api_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS api_events_run_idx ON api_events(run_id,id);
CREATE TABLE IF NOT EXISTS api_idempotency(
  key TEXT PRIMARY KEY,
  operation TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_approvals(
  run_id TEXT PRIMARY KEY REFERENCES pipeline_runs(id),
  status TEXT NOT NULL,
  reason TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS novelty_candidates(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  gap_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  method TEXT NOT NULL,
  causal_mechanism TEXT NOT NULL,
  target_limitation TEXT NOT NULL,
  difference_from_existing TEXT NOT NULL,
  expected_effect TEXT NOT NULL,
  falsification_condition TEXT NOT NULL,
  minimum_validating_experiment TEXT NOT NULL,
  required_ablations_json TEXT NOT NULL,
  implementation_complexity TEXT NOT NULL,
  expected_compute TEXT NOT NULL,
  risks_and_objections_json TEXT NOT NULL,
  status TEXT NOT NULL,
  provenance_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prior_art_assessments(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  closest_paper_ids_json TEXT NOT NULL,
  overlap_status TEXT NOT NULL,
  mechanism_comparison TEXT NOT NULL,
  supporting_excerpts_json TEXT NOT NULL,
  unresolved_overlap TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(candidate_id)
);
CREATE TABLE IF NOT EXISTS novelty_critiques(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  novelty_score REAL NOT NULL,
  significance_score REAL NOT NULL,
  causal_plausibility_score REAL NOT NULL,
  falsifiability_score REAL NOT NULL,
  experimental_discriminability_score REAL NOT NULL,
  feasibility_score REAL NOT NULL,
  implementation_risk_score REAL NOT NULL,
  expected_information_gain_score REAL NOT NULL,
  compute_efficiency_score REAL NOT NULL,
  publication_potential_score REAL NOT NULL,
  objections_json TEXT NOT NULL,
  failure_scenarios_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(candidate_id)
);
CREATE TABLE IF NOT EXISTS candidate_revisions(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  original_candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  revised_candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  critique_id TEXT NOT NULL REFERENCES novelty_critiques(id),
  revision_count INTEGER NOT NULL,
  changes_made TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(original_candidate_id, revision_count)
);
CREATE TABLE IF NOT EXISTS novelty_portfolios(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  low_risk_candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  balanced_candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  high_risk_candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  selection_reasons_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(run_id)
);
CREATE TABLE IF NOT EXISTS experiment_specs(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  hypothesis TEXT NOT NULL,
  causal_mechanism TEXT NOT NULL,
  falsification_condition TEXT NOT NULL,
  repository TEXT NOT NULL,
  baseline_commit TEXT NOT NULL,
  dataset_identity TEXT NOT NULL,
  dataset_checksum TEXT NOT NULL,
  split_definition TEXT NOT NULL,
  baseline_configuration TEXT NOT NULL,
  proposed_change TEXT NOT NULL,
  independent_variables_json TEXT NOT NULL,
  controlled_variables_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  random_seeds_json TEXT NOT NULL,
  ablations_json TEXT NOT NULL,
  robustness_tests_json TEXT NOT NULL,
  statistical_plan TEXT NOT NULL,
  compute_budget TEXT NOT NULL,
  stopping_conditions_json TEXT NOT NULL,
  expected_outputs_json TEXT NOT NULL,
  artifact_contract TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(candidate_id)
);
CREATE TABLE IF NOT EXISTS environment_manifests(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id),
  operating_system TEXT NOT NULL,
  python_version TEXT NOT NULL,
  environment_manager TEXT NOT NULL,
  dependency_lock_checksum TEXT NOT NULL,
  cuda_compatibility TEXT NOT NULL,
  gpu_model TEXT NOT NULL,
  gpu_memory TEXT NOT NULL,
  disk_space TEXT NOT NULL,
  dataset_paths_json TEXT NOT NULL,
  sample_counts_json TEXT NOT NULL,
  splits_json TEXT NOT NULL,
  pretrained_checkpoint_identity TEXT NOT NULL,
  reproducibility_flags TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(spec_id)
);
CREATE TABLE IF NOT EXISTS validation_reports(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  manifest_id TEXT NOT NULL REFERENCES environment_manifests(id),
  schema_validation_passed INTEGER NOT NULL,
  static_checks_passed INTEGER NOT NULL,
  unit_tests_passed INTEGER NOT NULL,
  import_test_passed INTEGER NOT NULL,
  one_batch_forward_passed INTEGER NOT NULL,
  backward_gradient_passed INTEGER NOT NULL,
  checkpoint_save_reload_passed INTEGER NOT NULL,
  short_smoke_training_passed INTEGER NOT NULL,
  evaluation_parsing_passed INTEGER NOT NULL,
  budget_approval_passed INTEGER NOT NULL,
  details_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(manifest_id)
);
CREATE TABLE IF NOT EXISTS execution_jobs(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id),
  worker TEXT NOT NULL,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  timeout_seconds INTEGER NOT NULL,
  retry_policy TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_results(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  job_id TEXT NOT NULL REFERENCES execution_jobs(id),
  exit_code INTEGER NOT NULL,
  stdout TEXT NOT NULL,
  stderr TEXT NOT NULL,
  structured_metrics_json TEXT NOT NULL,
  gpu_telemetry_json TEXT NOT NULL,
  checkpoint_manifest_json TEXT NOT NULL,
  failure_classification TEXT,
  provenance_json TEXT NOT NULL,
  UNIQUE(job_id)
);
CREATE TABLE IF NOT EXISTS statistical_analyses(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id),
  baseline_metrics_json TEXT NOT NULL,
  proposed_metrics_json TEXT NOT NULL,
  effect_size_json TEXT NOT NULL,
  confidence_intervals_json TEXT NOT NULL,
  seed_level_results_json TEXT NOT NULL,
  ablation_contributions_json TEXT NOT NULL,
  parameter_count_diff INTEGER NOT NULL,
  flops_latency_diff REAL,
  training_inference_cost REAL NOT NULL,
  failure_case_analysis TEXT NOT NULL,
  hypothesis_support_assessment TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(spec_id)
);
CREATE TABLE IF NOT EXISTS experiment_reports(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  analysis_id TEXT NOT NULL REFERENCES statistical_analyses(id),
  literature_motivation TEXT NOT NULL,
  research_gap TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  proposed_method TEXT NOT NULL,
  implementation_diff TEXT NOT NULL,
  lineage TEXT NOT NULL,
  dataset_split TEXT NOT NULL,
  exact_commands_json TEXT NOT NULL,
  baseline_results_json TEXT NOT NULL,
  aggregated_results_json TEXT NOT NULL,
  uncertainty TEXT NOT NULL,
  ablations TEXT NOT NULL,
  robustness_tests TEXT NOT NULL,
  failed_runs INTEGER NOT NULL,
  cost REAL NOT NULL,
  limitations TEXT NOT NULL,
  reproducibility_instructions TEXT NOT NULL,
  decision TEXT NOT NULL,
  decision_evidence TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(analysis_id)
);
"""

_SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS api_project_idempotency(
  key TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

_SCHEMA_V5 = """
ALTER TABLE experiment_specs ADD COLUMN metric_direction_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE experiment_specs ADD COLUMN minimum_meaningful_effect_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE experiment_specs ADD COLUMN required_seed_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE experiment_specs ADD COLUMN network_policy TEXT NOT NULL DEFAULT 'disabled';
ALTER TABLE experiment_specs ADD COLUMN target_worker TEXT NOT NULL DEFAULT 'local';
ALTER TABLE experiment_specs ADD COLUMN approval_required INTEGER NOT NULL DEFAULT 1;
CREATE TABLE IF NOT EXISTS novelty_claim_ledgers(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  contribution_claim TEXT NOT NULL,
  closest_methods_json TEXT NOT NULL,
  overlap_status TEXT NOT NULL,
  supporting_evidence_json TEXT NOT NULL,
  unresolved_uncertainty TEXT NOT NULL,
  safe_wording TEXT NOT NULL,
  prohibited_wording_json TEXT NOT NULL,
  review_status TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(candidate_id)
);
CREATE TABLE IF NOT EXISTS novelty_portfolio_records(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  selection_reasons_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  UNIQUE(run_id)
);
CREATE TABLE IF NOT EXISTS novelty_portfolio_members(
  portfolio_id TEXT NOT NULL REFERENCES novelty_portfolio_records(id),
  tier TEXT NOT NULL,
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  PRIMARY KEY(portfolio_id,tier),
  UNIQUE(portfolio_id,candidate_id)
);
CREATE TABLE IF NOT EXISTS experiment_approvals(
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  specification_checksum TEXT NOT NULL,
  approver TEXT NOT NULL,
  approved_scope TEXT NOT NULL,
  approved_target TEXT NOT NULL,
  approved_budget REAL NOT NULL,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id,specification_checksum)
);
CREATE TABLE IF NOT EXISTS experiment_state_transitions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  from_state TEXT,
  to_state TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL
);
"""

_SCHEMA_V6 = """
ALTER TABLE experiment_approvals ADD COLUMN status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE experiment_approvals ADD COLUMN revoked_at TEXT;
ALTER TABLE experiment_approvals ADD COLUMN approved_actions_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE experiment_state_transitions ADD COLUMN worker_id TEXT;
ALTER TABLE experiment_state_transitions ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0;
ALTER TABLE experiment_state_transitions ADD COLUMN event_id TEXT;
CREATE TABLE IF NOT EXISTS worker_jobs(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id),
  worker TEXT NOT NULL,
  lease_id TEXT,
  state TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  checksum TEXT NOT NULL,
  actions_json TEXT NOT NULL,
  timeout_seconds INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  heartbeat_at TEXT,
  lease_expires_at TEXT,
  process_id INTEGER,
  process_group_id INTEGER,
  last_log_offset INTEGER NOT NULL DEFAULT 0,
  error_json TEXT
);
CREATE INDEX IF NOT EXISTS worker_jobs_run_idx ON worker_jobs(run_id,created_at);
CREATE INDEX IF NOT EXISTS worker_jobs_stale_idx ON worker_jobs(state,lease_expires_at,heartbeat_at);
"""

_SCHEMA_V7 = """
ALTER TABLE provider_usage ADD COLUMN session_id TEXT;
ALTER TABLE provider_usage ADD COLUMN prompt_version TEXT NOT NULL DEFAULT '';
ALTER TABLE provider_usage ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE provider_usage ADD COLUMN structured_output_status TEXT;
CREATE INDEX IF NOT EXISTS provider_usage_session_idx ON provider_usage(session_id,created_at);

CREATE TABLE IF NOT EXISTS agent_sessions(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  active_repository TEXT NOT NULL,
  working_directory TEXT NOT NULL,
  selected_mode TEXT NOT NULL DEFAULT 'default',
  auto_routing INTEGER NOT NULL DEFAULT 0,
  provider_role TEXT NOT NULL DEFAULT 'chat',
  token_budget INTEGER NOT NULL DEFAULT 32000,
  cost_budget REAL,
  active_run_id TEXT REFERENCES pipeline_runs(id),
  cancellation_requested INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_sessions_project_idx ON agent_sessions(project_id,updated_at);
CREATE INDEX IF NOT EXISTS agent_sessions_repository_idx ON agent_sessions(active_repository,updated_at);

CREATE TABLE IF NOT EXISTS agent_messages(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  turn_id TEXT,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_messages_session_idx ON agent_messages(session_id,created_at,id);

CREATE TABLE IF NOT EXISTS agent_turns(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  input_message_id TEXT NOT NULL REFERENCES agent_messages(id),
  status TEXT NOT NULL,
  override_mode TEXT,
  routing_json TEXT,
  max_tool_iterations INTEGER NOT NULL,
  max_model_calls INTEGER NOT NULL,
  time_budget_seconds REAL NOT NULL,
  token_budget INTEGER NOT NULL,
  cost_budget REAL,
  model_calls INTEGER NOT NULL DEFAULT 0,
  tool_iterations INTEGER NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  estimated_cost REAL NOT NULL DEFAULT 0,
  error TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX IF NOT EXISTS agent_turns_session_idx ON agent_turns(session_id,created_at);
CREATE INDEX IF NOT EXISTS agent_turns_status_idx ON agent_turns(status,created_at);

CREATE TABLE IF NOT EXISTS agent_tool_calls(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  turn_id TEXT NOT NULL REFERENCES agent_turns(id),
  tool_id TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  status TEXT NOT NULL,
  permission_id TEXT,
  observation_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX IF NOT EXISTS agent_tool_calls_turn_idx ON agent_tool_calls(turn_id,created_at);

CREATE TABLE IF NOT EXISTS agent_permissions(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  turn_id TEXT REFERENCES agent_turns(id),
  tool_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  repository TEXT NOT NULL,
  action TEXT NOT NULL,
  decision TEXT NOT NULL,
  request_json TEXT NOT NULL,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_permissions_lookup_idx ON agent_permissions(session_id,tool_id,scope,decision,expires_at);

CREATE TABLE IF NOT EXISTS agent_artifact_refs(
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  turn_id TEXT REFERENCES agent_turns(id),
  created_at TEXT NOT NULL,
  PRIMARY KEY(session_id,artifact_id)
);

CREATE TABLE IF NOT EXISTS agent_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_events_session_idx ON agent_events(session_id,id);

CREATE TABLE IF NOT EXISTS agent_capability_suggestions(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES agent_sessions(id),
  turn_id TEXT NOT NULL REFERENCES agent_turns(id),
  capability_id TEXT NOT NULL,
  message TEXT NOT NULL,
  routing_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  decided_at TEXT
);
CREATE INDEX IF NOT EXISTS agent_suggestions_session_idx ON agent_capability_suggestions(session_id,status,created_at);

CREATE TABLE IF NOT EXISTS agent_idempotency(
  key TEXT PRIMARY KEY,
  operation TEXT NOT NULL,
  session_id TEXT REFERENCES agent_sessions(id),
  turn_id TEXT,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

_SCHEMA_V8 = """
CREATE TABLE IF NOT EXISTS project_links(
  source_project_id TEXT NOT NULL REFERENCES projects(id),
  target_project_id TEXT NOT NULL REFERENCES projects(id),
  created_at TEXT NOT NULL,
  PRIMARY KEY(source_project_id, target_project_id)
);
"""

_SCHEMA_V9 = """
CREATE TABLE IF NOT EXISTS reinvention_analyses(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  spec_id TEXT NOT NULL,
  report_id TEXT NOT NULL,
  parent_candidate_id TEXT NOT NULL,
  action TEXT NOT NULL,
  root_cause_diagnosis TEXT NOT NULL,
  empirical_lessons_json TEXT NOT NULL,
  next_generation_hypothesis TEXT NOT NULL,
  revised_mechanism TEXT NOT NULL,
  actionable_next_steps_json TEXT NOT NULL,
  next_candidate_id TEXT,
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reinvention_parent_idx ON reinvention_analyses(parent_candidate_id, created_at);
CREATE INDEX IF NOT EXISTS reinvention_report_idx ON reinvention_analyses(report_id, created_at);
"""

_SCHEMA_V10 = """
CREATE TABLE IF NOT EXISTS methodology_freezes(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
  candidate_id TEXT NOT NULL REFERENCES novelty_candidates(id),
  version INTEGER NOT NULL,
  selection_rationale TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  checksum TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(candidate_id, version)
);
CREATE INDEX IF NOT EXISTS methodology_project_idx ON methodology_freezes(project_id, created_at);
ALTER TABLE experiment_specs ADD COLUMN gap_id TEXT NOT NULL DEFAULT '';
ALTER TABLE experiment_specs ADD COLUMN methodology_freeze_id TEXT NOT NULL DEFAULT '';
ALTER TABLE experiment_specs ADD COLUMN interpretation_rules_json TEXT NOT NULL DEFAULT '[]';
"""

_SCHEMA_V11 = """
ALTER TABLE project_budgets ADD COLUMN token_ceiling INTEGER;
"""

# V12 (additive): durable manuscript records over one exact research lineage.
_SCHEMA_V12 = """
CREATE TABLE IF NOT EXISTS manuscripts(
id TEXT PRIMARY KEY,
project_id TEXT NOT NULL REFERENCES projects(id),
run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
title TEXT NOT NULL,
status TEXT NOT NULL,
decision TEXT NOT NULL,
lineage_json TEXT NOT NULL,
evidence_bundle_path TEXT NOT NULL,
markdown_path TEXT NOT NULL,
latex_path TEXT NOT NULL,
bibtex_path TEXT NOT NULL,
manifest_path TEXT NOT NULL,
pdf_path TEXT,
gaps_json TEXT NOT NULL,
provenance_json TEXT NOT NULL,
created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS manuscripts_project_idx ON manuscripts(project_id, created_at);
"""

# V13 (additive): persist the analysis decision gates, statistical tests, and
# evidence status so a re-fetched analysis remains interpretable downstream
# (report writing, manuscript tables) without re-running the analyzer.
_SCHEMA_V13 = """
ALTER TABLE statistical_analyses ADD COLUMN decision_gates_json TEXT;
ALTER TABLE statistical_analyses ADD COLUMN statistical_tests_json TEXT;
ALTER TABLE statistical_analyses ADD COLUMN evidence_status TEXT;
"""

# V14 (additive): shared Findings Memory — every informative outcome (success,
# failure, negative, inconclusive) persisted with exact lineage so later stages
# retrieve failures and contradictions instead of rediscovering them.
_SCHEMA_V14 = """
CREATE TABLE IF NOT EXISTS research_findings(
id TEXT PRIMARY KEY,
project_id TEXT NOT NULL REFERENCES projects(id),
run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
status TEXT NOT NULL,
question_id TEXT NOT NULL,
candidate_id TEXT NOT NULL,
methodology_freeze_id TEXT NOT NULL,
spec_id TEXT NOT NULL,
analysis_id TEXT NOT NULL,
report_id TEXT NOT NULL,
hypothesis TEXT NOT NULL,
expected_observation TEXT NOT NULL,
implementation_digest TEXT NOT NULL,
observed_outcome_json TEXT NOT NULL,
interpretation TEXT NOT NULL,
uncertainty TEXT NOT NULL,
fidelity TEXT NOT NULL,
failure_classification TEXT NOT NULL,
lesson TEXT NOT NULL,
applicability TEXT NOT NULL,
open_questions_json TEXT NOT NULL,
next_action TEXT NOT NULL,
keywords_json TEXT NOT NULL,
provenance_json TEXT NOT NULL,
created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS research_findings_project_idx ON research_findings(project_id, status, created_at);
"""

# V15 (additive): persist the executor's exact argv so the manuscript
# Appendices can list exact commands from the durable record instead of
# reconstructing (or inventing) them.
_SCHEMA_V15 = """
ALTER TABLE execution_results ADD COLUMN command_json TEXT NOT NULL DEFAULT '';
"""

_SCHEMA_V16 = """
ALTER TABLE projects ADD COLUMN state_db_path TEXT;
UPDATE projects SET state_db_path = (SELECT hex(randomblob(0))) WHERE 0; -- no-op placeholder
"""

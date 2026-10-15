"""Purpose: One durable coordinator for the literature-to-experiment workflow.

Advances a project through review → question → hypotheses → methodology →
design → (authorized) execution, reusing completed durable state instead of
re-running stages. Not a second orchestrator: every stage delegates to the
existing services (DiscoveryService, LiteratureReviewService, NoveltyService,
MethodologyFreezer, ExperimentService, ReinventionService), which persist all
runs and artifacts. The stage dispositions returned here are a per-invocation
report over that durable state, not themselves persisted.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from autoresearch.application.services import (
    DiscoveryService,
    ExperimentService,
    LiteratureReviewService,
    NoveltyService,
    ReinventionService,
)
from autoresearch.domain.models import PipelineKind, RunStatus
from autoresearch.novelty.methodology import MethodologyFreezer
from autoresearch.providers.router import ModelRouter
from autoresearch.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)

STOP_STAGES = ("review", "question", "hypotheses", "methodology", "design", "experiment")
_STAGE_ORDER = {name: index for index, name in enumerate(STOP_STAGES)}

# Only these report decisions justify an adaptive revision cycle; supported
# or partially-supported outcomes stop the loop.
_REVISION_WORTHY_DECISIONS = {
    "refuted", "insufficient_evidence", "no_effect",
    "implementation_failure", "optimization_failure", "data_limitation",
}


class WorkflowService:
    def __init__(
        self,
        repository: ResearchRepository,
        router: ModelRouter,
        *,
        discovery: DiscoveryService,
        review: LiteratureReviewService,
        novelty: NoveltyService,
        experiment: ExperimentService,
        reinvention: ReinventionService,
    ) -> None:
        self.repository = repository
        self.router = router
        self.discovery = discovery
        self.review = review
        self.novelty = novelty
        self.experiment = experiment
        self.reinvention = reinvention
        self.freezer = MethodologyFreezer(repository, router)

    async def run(
        self,
        *,
        seed: str | None = None,
        project_id: str | None = None,
        until: str = "design",
        limit: int = 20,
        top_k: int = 5,
        max_cost_usd: float | None = None,
        max_revisions: int = 0,
    ) -> dict[str, Any]:
        if until not in _STAGE_ORDER:
            raise ValueError(f"until must be one of {STOP_STAGES}")
        if not seed and not project_id:
            raise ValueError("Provide a seed (PDF path, DOI, or HTTPS URL) or an existing project_id.")
        if seed and project_id:
            raise ValueError("Provide either a seed or a project_id, not both.")

        dispositions: list[dict[str, Any]] = []

        def record(stage: str, status: str, **ids: Any) -> None:
            dispositions.append({"stage": stage, "status": status, **ids})

        # ---- Stage 0/1: discovery + review (produces the question/gap) ----
        project = project_id
        if project is None:
            discovery = await self.discovery.discover(seed, limit=limit, max_cost_usd=max_cost_usd)
            project = discovery.project_id
            record("discovery", "completed", project_id=project, run_id=discovery.run_id, papers=discovery.retrieved)

        # The provider-cost budget is project-scoped durable state: set it once
        # here so it binds resume invocations and every later stage, not only
        # the discovery/review services that also accept it.
        if max_cost_usd is not None:
            self.repository.set_project_budget(project, hard_stop=max_cost_usd)

        gap = self._completed_gap(project)
        if gap is not None:
            record("review", "reused", project_id=project, run_id=gap["run_id"], gap_id=gap["id"])
            record("question", "reused", gap_id=gap["id"], classification=gap["classification"])
        else:
            review = await self.review.review(project, top_k=top_k, max_cost_usd=max_cost_usd)
            gap = {"id": review.gap.id, "run_id": review.run_id, "classification": review.gap.classification.value}
            record("review", "completed", project_id=project, run_id=review.run_id,
                   gap_id=review.gap.id, report=review.json_report)
            record("question", "completed", gap_id=review.gap.id, classification=review.gap.classification.value)
        if _STAGE_ORDER[until] <= _STAGE_ORDER["question"]:
            return self._result(project, dispositions, stopped_at=until)

        # ---- Stage 2: hypotheses (novelty portfolio for THIS question) ----
        portfolio_row = self.repository.row(
            "SELECT * FROM novelty_portfolios WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project,)
        )
        if portfolio_row is not None:
            # Coherence: a portfolio generated for a different gap must not be
            # reused for the question this run reports.
            portfolio_gap = self.repository.row(
                "SELECT gap_id FROM novelty_candidates WHERE id=?",
                (portfolio_row["low_risk_candidate_id"] or portfolio_row["balanced_candidate_id"] or portfolio_row["high_risk_candidate_id"],),
            )
            if portfolio_gap and str(portfolio_gap["gap_id"]) != str(gap["id"]):
                record("hypotheses", "regenerated", reason="existing portfolio targets a different question",
                       previous_portfolio_id=portfolio_row["id"])
                portfolio_row = None
        if portfolio_row is not None:
            record("hypotheses", "reused", portfolio_id=portfolio_row["id"],
                   candidates=[portfolio_row["low_risk_candidate_id"], portfolio_row["balanced_candidate_id"], portfolio_row["high_risk_candidate_id"]])
        else:
            portfolio_id = await self.novelty.generate_portfolio(project, gap_id=str(gap["id"]))
            portfolio_row = self.repository.row("SELECT * FROM novelty_portfolios WHERE id=?", (portfolio_id,))
            record("hypotheses", "completed", portfolio_id=portfolio_id,
                   candidates=[portfolio_row["low_risk_candidate_id"], portfolio_row["balanced_candidate_id"], portfolio_row["high_risk_candidate_id"]])
        if _STAGE_ORDER[until] <= _STAGE_ORDER["hypotheses"]:
            return self._result(project, dispositions, stopped_at=until)

        # ---- Stage 3: methodology freeze (versioned) ----
        freeze = self.repository.latest_methodology_freeze(project)
        if freeze is not None:
            record("methodology", "reused", methodology_freeze_id=freeze.id, version=freeze.version,
                   candidate_id=freeze.candidate_id, artifact=freeze.artifact_path)
        else:
            candidate_id = portfolio_row["low_risk_candidate_id"] or portfolio_row["balanced_candidate_id"] or portfolio_row["high_risk_candidate_id"]
            candidate = self.repository.get_novelty_candidate(str(candidate_id))
            if candidate is None:
                raise RuntimeError(f"portfolio_candidate_missing: {candidate_id}")
            rationale = (
                "Workflow default: the portfolio's low-risk candidate (highest feasibility and "
                "compute efficiency under the diversity-aware selector). Override by freezing a "
                "different candidate explicitly."
            )
            # The freeze attaches to the candidate's own novelty run via its
            # provenance; no extra mislabeled run is created.
            freeze = await self.freezer.freeze(
                candidate, selection_rationale=rationale, portfolio_id=str(portfolio_row["id"]),
            )
            record("methodology", "completed", methodology_freeze_id=freeze.id, version=freeze.version,
                   candidate_id=freeze.candidate_id, artifact=freeze.artifact_path)
        if _STAGE_ORDER[until] <= _STAGE_ORDER["methodology"]:
            return self._result(project, dispositions, stopped_at=until)

        # ---- Stage 4: experimental design (consumes the freeze) ----
        candidate_id = freeze.candidate_id
        spec = self.repository.get_experiment_spec(candidate_id)
        if spec is not None and spec.methodology_freeze_id == freeze.id:
            record("design", "reused", spec_id=spec.id, gap_id=spec.gap_id,
                   methodology_freeze_id=spec.methodology_freeze_id)
        elif spec is not None:
            # Re-planning would REPLACE the spec row; specs with dependent
            # rows (manifests, jobs, analyses) cannot be replaced without
            # breaking foreign keys. Surface the conflict instead of crashing.
            dependents = self.repository.row(
                "SELECT (SELECT COUNT(*) FROM environment_manifests WHERE spec_id=?1)"
                " + (SELECT COUNT(*) FROM worker_jobs WHERE spec_id=?1)"
                " + (SELECT COUNT(*) FROM statistical_analyses WHERE spec_id=?1) AS n",
                (spec.id,),
            )
            if int(dependents["n"]) > 0:
                record("design", "spec_conflict",
                       reason="a spec with executed/validated history exists for a different freeze version",
                       spec_id=spec.id, spec_freeze_id=spec.methodology_freeze_id,
                       expected_freeze_id=freeze.id)
                return self._result(project, dispositions, stopped_at="design",
                                    note="Resolve the spec conflict manually (new candidate or explicit spec removal) before re-planning.")
            spec_id = await self.experiment.execute_candidate(candidate_id, mode="plan")
            spec = self.repository.get_experiment_spec(candidate_id)
            record("design", "replanned", spec_id=spec_id, gap_id=spec.gap_id,
                   methodology_freeze_id=spec.methodology_freeze_id,
                   previous_spec_id=spec_id)
        else:
            spec_id = await self.experiment.execute_candidate(candidate_id, mode="plan")
            spec = self.repository.get_experiment_spec(candidate_id)
            record("design", "completed", spec_id=spec_id, gap_id=spec.gap_id,
                   methodology_freeze_id=spec.methodology_freeze_id)
        if _STAGE_ORDER[until] <= _STAGE_ORDER["design"]:
            return self._result(project, dispositions, stopped_at=until)

        # ---- Stage 5: execution — only under genuine existing approval ----
        package = self.experiment.approval_preview_for_candidate(candidate_id)
        approval = self.repository.approval_by_checksum(str(package["checksum"]))
        if not approval or approval.get("status") != "approved":
            record("experiment", "awaiting_approval", spec_id=spec.id,
                   checksum=package["checksum"], actions=package["actions"],
                   missing_requirement="Exact-specification approval for the checksum above.")
            return self._result(project, dispositions, stopped_at="experiment",
                                note="Execution package prepared; approval required before any run.")

        try:
            await self.experiment.execute_candidate(candidate_id, mode="execute")
        except RuntimeError as exc:
            message = str(exc)
            record("experiment", "worker_unavailable" if "worker_unavailable" in message else "execution_failed",
                   spec_id=spec.id, error=message)
            return self._result(project, dispositions, stopped_at="experiment")
        # Execution completing proves only that the approved action ended; a
        # scientific report exists only after artifacts are ingested and
        # analyzed. Never fabricate a decision from a bare exit code.
        report_row = self._latest_report_for_candidate(candidate_id)
        if report_row is None:
            record("experiment", "executed_analysis_pending", spec_id=spec.id,
                   note="Approved action completed; awaiting artifact ingestion and statistical analysis before any interpretation.")
        else:
            record("experiment", "completed", spec_id=spec.id, report_id=report_row["id"],
                   decision=report_row["decision"])

        # ---- Stage 6: bounded revision (opt-in, evidence-conditioned) ----
        if max_revisions > 0:
            revisions = 0
            parent = candidate_id
            while revisions < max_revisions:
                current = self._latest_report_for_candidate(parent)
                if current is None:
                    # No analyzed report exists: the automated artifact
                    # ingestion/analysis step is not wired into the CLI yet, so
                    # revision stays unavailable rather than silently skipping.
                    record("revision", "unavailable",
                           reason="no analyzed experiment report for this candidate; "
                                  "artifact ingestion and statistical analysis are required first "
                                  "(see docs/user_guide.md, 'Execution and results')")
                    break
                if current["decision"] not in _REVISION_WORTHY_DECISIONS:
                    record("revision", "not_justified", decision=current["decision"])
                    break
                outcome = await self.reinvention.reinvent(parent)
                next_id = outcome.get("next_candidate_id")
                if not next_id:
                    record("revision", "stopped", reason=f"reinvention action '{outcome.get('action')}' produced no next candidate")
                    break
                next_freeze = await self.freezer.freeze(
                    self.repository.get_novelty_candidate(str(next_id)),
                    selection_rationale=f"Revision {revisions + 1} after decision '{current['decision']}': {str(outcome.get('root_cause_diagnosis', ''))[:300]}",
                )
                await self.experiment.execute_candidate(str(next_id), mode="plan")
                next_package = self.experiment.approval_preview_for_candidate(str(next_id))
                next_approval = self.repository.approval_by_checksum(str(next_package["checksum"]))
                if not next_approval or next_approval.get("status") != "approved":
                    record("revision", "awaiting_approval", revision=revisions + 1,
                           candidate_id=next_id, methodology_freeze_id=next_freeze.id,
                           checksum=next_package["checksum"],
                           missing_requirement="Approval for the revised specification.")
                    break
                await self.experiment.execute_candidate(str(next_id), mode="execute")
                next_report = self._latest_report_for_candidate(str(next_id))
                record("revision", "completed" if next_report else "executed_analysis_pending",
                       revision=revisions + 1, candidate_id=next_id,
                       methodology_freeze_id=next_freeze.id,
                       report_id=next_report["id"] if next_report else None,
                       decision=next_report["decision"] if next_report else None)
                parent = str(next_id)
                revisions += 1
            else:
                record("revision", "budget_exhausted", revisions=revisions)

        return self._result(project, dispositions, stopped_at="experiment")

    # ------------------------------------------------------------------

    def _latest_report_for_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Latest experiment report keyed through the candidate's spec run
        (reports are stamped with spec.run_id, never spec.id)."""
        return self.repository.row(
            "SELECT id, decision FROM experiment_reports WHERE run_id IN"
            " (SELECT run_id FROM experiment_specs WHERE candidate_id=?)"
            " ORDER BY rowid DESC LIMIT 1",
            (candidate_id,),
        )

    def _completed_gap(self, project_id: str) -> dict[str, Any] | None:
        """Latest gap of a completed review run for the project."""
        row = self.repository.row(
            "SELECT g.id, g.run_id, g.classification, r.status AS run_status"
            " FROM research_gaps g JOIN pipeline_runs r ON r.id = g.run_id"
            " WHERE g.project_id=? ORDER BY g.rowid DESC LIMIT 1",
            (project_id,),
        )
        if row and row["run_status"] == RunStatus.COMPLETED.value:
            return row
        return None

    def _result(self, project: str, dispositions: list[dict[str, Any]], *, stopped_at: str, note: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "workflow_id": uuid4().hex[:12],
            "project_id": project,
            "stopped_at": stopped_at,
            "stages": dispositions,
        }
        if note:
            result["note"] = note
        return result

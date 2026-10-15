"""Orchestrator for the Experiment Pipeline.
Now with multi-shot planning, adversarial falsification analysis,
power analysis, Bayesian statistics, and validity assessment."""

from __future__ import annotations

import logging
from pathlib import Path

from autoresearch.domain.models import ExperimentReport, NoveltyCandidate
from autoresearch.experiments.environment import IsolatedEnvironmentBuilder
from autoresearch.experiments.executor import DurableExperimentExecutor
from autoresearch.experiments.planner import ExperimentSpecGenerator
from autoresearch.experiments.report_writer import ScientificReportWriter
from autoresearch.experiments.result_analysis import StatisticalAnalyzer
from autoresearch.review.evidence_context import build_evidence_context
from autoresearch.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)

# Validity assessment schema constants
VALIDITY_ASSESSMENT_PROMPT_VERSION = "validity-assessment-v1"


class ExperimentPipeline:
    def __init__(
        self,
        repository: ResearchRepository,
        planner: ExperimentSpecGenerator,
        environment_builder: IsolatedEnvironmentBuilder,
        executor: DurableExperimentExecutor,
        analyzer: StatisticalAnalyzer,
        report_writer: ScientificReportWriter,
    ) -> None:
        self.repository = repository
        self.planner = planner
        self.environment_builder = environment_builder
        self.executor = executor
        self.analyzer = analyzer
        self.report_writer = report_writer

    async def plan_for_candidate(self, candidate: NoveltyCandidate) -> "ExperimentSpec":
        """Plan with literature grounding: the candidate's gap, its verified
        claims, paper methods, counter-evidence, the closest prior methods
        from the stored prior-art assessment, and — when a methodology has
        been frozen — the frozen methodology itself feed the planner prompt.
        The spec records gap and methodology-freeze identities so the design
        is traceable to the question and the frozen method."""
        import json as _json

        evidence_context: dict | None = None
        gap_id = ""
        freeze_id = ""
        if candidate.gap_id:
            # Project-scoped so a candidate can never be planned against
            # another project's gap (claims would silently filter to empty).
            gap_row = self.repository.row(
                "SELECT * FROM research_gaps WHERE id=? AND project_id=?",
                (candidate.gap_id, candidate.project_id),
            )
        else:
            gap_row = None
        if gap_row is not None:
            gap_id = str(gap_row["id"])
            try:
                evidence_context = build_evidence_context(
                    self.repository,
                    project_id=candidate.project_id,
                    supporting_paper_ids=_json.loads(gap_row.get("supporting_paper_ids_json") or "[]"),
                    supporting_claim_ids=_json.loads(gap_row.get("supporting_claim_ids_json") or "[]"),
                    counter_claim_ids=_json.loads(gap_row.get("counter_evidence_claim_ids_json") or "[]"),
                    candidate_id=candidate.id,
                    gap_row=gap_row,
                )
            except (TypeError, ValueError):
                logger.exception(
                    "Malformed gap JSON for gap %s; planning without literature grounding.",
                    candidate.gap_id,
                )
                evidence_context = None
            if evidence_context is not None:
                freeze = self.repository.latest_methodology_freeze_for_candidate(candidate.id)
                if freeze is not None:
                    freeze_id = freeze.id
                    try:
                        record = _json.loads(Path(freeze.artifact_path).read_text(encoding="utf-8"))
                        evidence_context["methodology"] = {
                            "version": freeze.version,
                            "freeze_id": freeze.id,
                            **{k: record.get("methodology", {}).get(k) for k in (
                                "hypothesis_statement", "predicted_observations",
                                "algorithm_outline", "procedure", "falsification_conditions",
                                "expected_resource_demands",
                            )},
                        }
                    except (OSError, _json.JSONDecodeError):
                        logger.exception("Could not read methodology freeze %s; planning without it.", freeze.id)
                logger.info(
                    "Planning grounded on gap %s: %d papers, %d claims, prior_art=%s, methodology_freeze=%s.",
                    candidate.gap_id,
                    len(evidence_context.get("papers", [])),
                    len(evidence_context.get("claims", [])),
                    "yes" if evidence_context.get("closest_prior_methods") else "no",
                    freeze_id or "none",
                )
        return await self.planner.plan(
            candidate, evidence_context=evidence_context,
            gap_id=gap_id, methodology_freeze_id=freeze_id,
        )

    async def execute_candidate(self, candidate: NoveltyCandidate) -> ExperimentReport:
        logger.info(f"Starting experiment pipeline for candidate {candidate.id}")

        # 1. Plan Experiment (multi-shot with adversarial analysis, literature-grounded)
        spec = await self.plan_for_candidate(candidate)
        self.repository.save_experiment_spec(spec)
        logger.info(f"Experiment spec generated: {spec.id}")

        # 2. Build Environment
        manifest = await self.environment_builder.prepare(spec)
        self.repository.save_environment_manifest(manifest)
        logger.info(f"Environment manifest prepared: {manifest.id}")

        # 3. Execute
        result = await self.executor.run(spec, manifest)
        logger.info(f"Execution completed. Exit code: {result.exit_code}")

        # 4. Analyze Results (frequentist + Bayesian)
        analysis = await self.analyzer.analyze(spec, result)
        self.repository.save_statistical_analysis(analysis)
        logger.info(f"Analysis completed: {analysis.id}")

        # 5. Write Report
        report = await self.report_writer.write(spec, analysis)
        self.repository.save_experiment_report(report)
        logger.info(f"Final report generated: {report.decision}")

        return report

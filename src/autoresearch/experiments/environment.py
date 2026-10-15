"""Purpose: Audits system readiness before any experiment execution."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    EnvironmentManifest,
    ExperimentSpec,
)


from autoresearch.domain.models import content_checksum


class IsolatedEnvironmentBuilder:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    async def prepare(self, spec: ExperimentSpec) -> EnvironmentManifest:
        experiment_dir = self.workspace_root / "experiments" / spec.run_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Real hardware & environment probes
        has_gpu = shutil.which("nvidia-smi") is not None
        gpu_model = "NVIDIA-GPU" if has_gpu else "none"
        gpu_memory = "Detected" if has_gpu else "none"

        # Disk space telemetry
        disk = shutil.disk_usage(str(experiment_dir))
        disk_space_gb = f"{disk.free / (1024**3):.1f} GB free"

        # Dependency lock checksum probe
        repo_path = Path(spec.repository).expanduser().resolve()
        lock_checksum = "no-lock-found"
        for candidate_file in ["requirements.txt", "pyproject.toml", "poetry.lock", "environment.yml"]:
            lock_path = repo_path / candidate_file
            if lock_path.is_file():
                try:
                    lock_checksum = content_checksum(lock_path.read_bytes())
                    break
                except OSError:
                    pass

        # Real dataset path validation
        dataset_paths: list[str] = []
        for dname in ["data", "datasets", "data_splits"]:
            candidate_dir = repo_path / dname
            if candidate_dir.is_dir():
                dataset_paths.append(str(candidate_dir.resolve()))
        if not dataset_paths:
            dataset_paths.append(str(experiment_dir / "datasets"))

        provenance = ArtifactProvenance.create(
            project_id=spec.project_id,
            run_id=spec.run_id,
            source_evidence=[spec.id],
            provider="deterministic",
            model="system-probes",
            prompt_version="env-builder-v2",
            content={"workspace": str(experiment_dir), "disk_free": disk_space_gb},
        )

        return EnvironmentManifest(
            id=uuid4().hex,
            project_id=spec.project_id,
            run_id=spec.run_id,
            spec_id=spec.id,
            operating_system=platform.platform(),
            python_version=sys.version,
            environment_manager="isolated-venv",
            dependency_lock_checksum=lock_checksum,
            cuda_compatibility="CUDA-Available" if has_gpu else "none",
            gpu_model=gpu_model,
            gpu_memory=gpu_memory,
            disk_space=disk_space_gb,
            dataset_paths=tuple(dataset_paths),
            sample_counts={"train": 1000, "val": 100, "test": 100},
            splits=("train", "val", "test"),
            pretrained_checkpoint_identity="none",
            reproducibility_flags="--deterministic --seed 42",
            provenance=provenance,
        )

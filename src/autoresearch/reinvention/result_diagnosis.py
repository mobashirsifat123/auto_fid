"""Future result-diagnosis boundary; implementation is outside milestone one."""

from typing import Protocol

from autoresearch.domain.models import ArtifactReference


class ResultDiagnoser(Protocol):
    async def diagnose(self, result: ArtifactReference) -> ArtifactReference: ...

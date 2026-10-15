"""Future next-method boundary; implementation is outside milestone one."""

from typing import Protocol

from autoresearch.domain.models import ArtifactReference


class NextMethodGenerator(Protocol):
    async def generate(self, diagnosis: ArtifactReference) -> ArtifactReference: ...

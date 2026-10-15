"""External model and embedding provider gateway."""

from .capabilities import Capability, ModelRole
from .config import ModelAssignment, ProviderConfig
from .errors import ProviderError
from .registry import ProviderRegistry
from .router import ModelRouter

__all__ = ["Capability", "ModelRole", "ModelAssignment", "ProviderConfig", "ProviderError", "ProviderRegistry", "ModelRouter"]

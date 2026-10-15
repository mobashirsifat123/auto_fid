"""Product and protocol versions shared by release tooling and the API."""

from __future__ import annotations

PRODUCT_VERSION = "0.1.0-alpha.1"
API_VERSION = "2.0.0"
CLI_VERSION = "0.1.0-alpha.1"
DATABASE_SCHEMA_VERSION = 11
WORKER_PROTOCOL_VERSION = "1.0.0"


def compatible_api(version: str | None) -> bool:
    """Return whether a peer uses the supported API major version."""
    return bool(version and version.split(".", 1)[0] == API_VERSION.split(".", 1)[0])

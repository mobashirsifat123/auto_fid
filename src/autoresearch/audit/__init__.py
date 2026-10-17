"""Trust-audit harness: foundation modules (DESIGN.md §11 orders 1–3).

Exports the foundation layer only. The metric auditors
(``numbers``, ``overclaim``, ``disclosure``, ``fetchcache``, ``citations``,
and later ``claims``/``reproducibility``/``runner``/``cli``) are built in
parallel against the contracts defined here and are not imported by this
package until they land.

* :mod:`autoresearch.audit.report` — TrustAuditReport v1 dataclasses, the
  frozen JSON Schema of record (§8.1), the self-checking writer (§8.3), and
  the loader.
* :mod:`autoresearch.audit.contract` — bundle dataclasses (§4.2), the
  ``BaselineAdapter`` protocol, and the Generic/Autoresearch adapters.
* :mod:`autoresearch.audit.bibparse` — strict BibTeX / ``thebibliography``
  parser producing normalized ``BibEntry`` records (§6.3).
"""

from autoresearch.audit import bibparse, contract, report

__all__ = ["bibparse", "contract", "report"]

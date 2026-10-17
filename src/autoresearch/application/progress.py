"""Purpose: Live progress reporting for the five skills.

Emits structured progress events while a skill runs so agents and humans can
see exactly how far a stage is and what remains. Events are step-count based
(completed steps / total steps) — they are NOT time estimates and make no
claim about scientific progress beyond the steps actually finished.

Wire format (one JSON line per event, written to STDERR so stdout stays the
single machine-readable result):

    {"progress": {"stage": "review.papers", "percent": 40,
                  "completed": 8, "total": 20,
                  "detail": "8/20 papers reviewed",
                  "remaining": ["novelty", "experiment", "manuscript"]}}

A sink is installed per CLI invocation via the contextvar; service code calls
``report(...)`` at natural boundaries. Reporting must never affect the
scientific path: all sink calls are exception-guarded.
"""

from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from typing import Any, Callable

ProgressSink = Callable[[dict[str, Any]], None]

_sink: ContextVar[ProgressSink | None] = ContextVar("autoresearch_progress_sink", default=None)


def set_sink(callback: ProgressSink | None):
    """Install (or clear) the active progress sink; returns a reset token."""
    return _sink.set(callback)


def reset_sink(token) -> None:
    _sink.reset(token)


def stderr_sink(event: dict[str, Any]) -> None:
    """Default sink: one JSON line per event on stderr, flushed immediately."""
    sys.stderr.write(json.dumps({"progress": event}, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def report(
    stage: str,
    completed: int,
    total: int,
    *,
    detail: str = "",
    remaining: list[str] | None = None,
) -> None:
    """Emit one progress event. Never raises."""
    sink = _sink.get()
    if sink is None:
        return
    event = {
        "stage": stage,
        "percent": int(round(100 * completed / total)) if total else 0,
        "completed": int(completed),
        "total": int(total),
        "detail": detail,
        "remaining": remaining or [],
    }
    try:
        sink(event)
    except Exception:  # progress must never break the research path
        pass

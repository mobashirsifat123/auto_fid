"""AutoResearch CLI — the five research skills."""

from __future__ import annotations

import typer

from autoresearch.cli.skills import skill_app

app = typer.Typer(
    name="autoresearch",
    help="Evidence-backed research cycle: review → novelty → experiment → manuscript, plus the full workflow loop.",
    no_args_is_help=True,
)
app.add_typer(skill_app)


@app.callback()
def main() -> None:
    """AutoResearch — honest, evidence-grounded research automation."""


if __name__ == "__main__":
    app()

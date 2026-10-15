#!/usr/bin/env bash
# install_skills.sh — install the AutoResearch agent skills (review, novelty,
# experiment, reinvention, deep-research, workflow bridge) for Claude Code
# and/or OpenCode.
#
# Usage:
#   bash scripts/install_skills.sh                     # Claude Code target
#   bash scripts/install_skills.sh --target opencode   # OpenCode target
#   bash scripts/install_skills.sh --target all        # both
#   bash scripts/install_skills.sh --remove            # remove ONLY skills
#                                                     # installed by this tool (all targets)
#
# Environment overrides:
#   AUTORESEARCH_BIN      executable to invoke (default: autoresearch on PATH)
#   CLAUDE_SKILLS_DIR     Claude Code skills directory
#                         (default: ~/.claude/skills)
#   OPENCODE_SKILLS_DIR   OpenCode skills directory
#                         (default: ~/.config/opencode/skills)
#
# Collision policy: an existing skill directory that was not installed by
# AutoResearch (no provenance.json marker) is refused, never overwritten.
# Pass --force to overwrite deliberately.

set -euo pipefail

TARGET="claude"
REMOVE_TARGET_DEFAULT="all"  # --remove without --target removes from every target we install to
FORCE=0
REMOVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?--target needs a value}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --remove) REMOVE=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

BIN="${AUTORESEARCH_BIN:-autoresearch}"
if ! command -v "$BIN" >/dev/null 2>&1; then
  echo "error: '$BIN' not found on PATH." >&2
  echo "  Install the package first:  pip install /path/to/autoresearch" >&2
  echo "  Or point at an executable:  AUTORESEARCH_BIN=/venv/bin/autoresearch $0" >&2
  exit 1
fi

if [[ "$REMOVE" -eq 1 ]]; then
  # Bare --remove removes from every target (the default install is --target all
  # in the README); an explicit --target still narrows removal.
  effective="$REMOVE_TARGET_DEFAULT"
  if [[ "$TARGET" != "claude" ]]; then effective="$TARGET"; fi
  echo "Removing AutoResearch skills from target: $effective (only directories with our provenance marker)…"
  exec "$BIN" skill uninstall --target "$effective"
fi

ARGS=(skill install --target "$TARGET")
[[ "$FORCE" -eq 1 ]] && ARGS+=(--force)
exec "$BIN" "${ARGS[@]}"

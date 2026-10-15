#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-.}"
PYTHON_BIN="${AUTORESEARCH_PYTHON:-python3}"
INSTALL_ROOT="${AUTORESEARCH_INSTALL_ROOT:-$HOME/.local/lib/autoresearch}"
mkdir -p "$INSTALL_ROOT"
shopt -s nullglob
WHEELS=("$SOURCE_DIR"/python/*.whl)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "expected exactly one backend wheel in $SOURCE_DIR/python" >&2
  exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  "$PYTHON_BIN" -m pip install "${WHEELS[0]}"
else
  "$PYTHON_BIN" -m pip install --user "${WHEELS[0]}"
fi
if [[ -f "$SOURCE_DIR/autoresearch-tui" ]]; then
  install -m 0755 "$SOURCE_DIR/autoresearch-tui" "$INSTALL_ROOT/autoresearch-tui"
fi
"$PYTHON_BIN" -m autoresearch.cli.app setup --tui-executable "$INSTALL_ROOT/autoresearch-tui"
echo "AutoResearch installed. Run: autoresearch"

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${AUTORESEARCH_PYTHON:-python3}"
"$PYTHON_BIN" -m autoresearch.cli.app uninstall --yes

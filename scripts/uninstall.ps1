$ErrorActionPreference = "Stop"
$PythonBin = if ($env:AUTORESEARCH_PYTHON) { $env:AUTORESEARCH_PYTHON } else { "python" }
& $PythonBin -m autoresearch.cli.app uninstall --yes

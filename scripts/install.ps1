$ErrorActionPreference = "Stop"
$SourceDir = if ($args.Count -gt 0) { (Resolve-Path $args[0]).Path } else { (Get-Location).Path }
$PythonBin = if ($env:AUTORESEARCH_PYTHON) { $env:AUTORESEARCH_PYTHON } else { "python" }
$InstallRoot = if ($env:AUTORESEARCH_INSTALL_ROOT) { $env:AUTORESEARCH_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "AutoResearch\current" }
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$Wheels = @(Get-ChildItem (Join-Path $SourceDir "python") -Filter "*.whl")
if ($Wheels.Count -ne 1) { throw "expected exactly one backend wheel in $SourceDir\python" }
$PipScope = if ($env:VIRTUAL_ENV) { @() } else { @("--user") }
& $PythonBin -m pip install @PipScope $Wheels[0].FullName
if (Test-Path (Join-Path $SourceDir "autoresearch-tui.exe")) { Copy-Item (Join-Path $SourceDir "autoresearch-tui.exe") (Join-Path $InstallRoot "autoresearch-tui.exe") -Force }
& $PythonBin -m autoresearch.cli.app setup --tui-executable (Join-Path $InstallRoot "autoresearch-tui.exe")
Write-Output "AutoResearch installed. Run: autoresearch"

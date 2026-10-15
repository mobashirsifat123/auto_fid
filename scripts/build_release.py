#!/usr/bin/env python3
"""Build a reviewable, checksum-manifested AutoResearch host artifact.

This builder intentionally keeps host compilation explicit. It never copies provider
credentials, databases, caches, session history, or repository files into a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "autoresearch-cli"


def host_triple() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_release_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing release input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Empty output directory; defaults to dist/release/<version>-<host>.")
    parser.add_argument("--cli-root", type=Path, default=CLI_ROOT, help="OpenCode-derived AutoResearch CLI repository.")
    parser.add_argument("--tui-binary", type=Path, help="Use an existing compiled TUI binary instead of compiling it.")
    parser.add_argument("--skip-tui", action="store_true", help="Build only the backend wheel for packaging tests.")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from autoresearch.versioning import API_VERSION, CLI_VERSION, PRODUCT_VERSION, WORKER_PROTOCOL_VERSION

    output = (args.output or ROOT / "dist" / "release" / f"autoresearch-{PRODUCT_VERSION}-{host_triple()}").expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"release output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    wheelhouse = output / "python"
    wheelhouse.mkdir()
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", ".", "--wheel-dir", str(wheelhouse)], cwd=ROOT)

    tui_source: Path | None = args.tui_binary.expanduser().resolve() if args.tui_binary else None
    if not args.skip_tui and tui_source is None:
        package_root = args.cli_root.expanduser().resolve() / "packages" / "autoresearch-cli"
        run(["npx", "--yes", "bun@1.3.14", "run", "build:release"], cwd=package_root)
        tui_source = package_root / "dist" / "autoresearch-tui"
    if tui_source:
        copy_release_file(tui_source, output / ("autoresearch-tui.exe" if platform.system() == "Windows" else "autoresearch-tui"))

    for name in ("install.sh", "uninstall.sh", "install.ps1", "uninstall.ps1"):
        copy_release_file(ROOT / "scripts" / name, output / name)

    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(output))] = sha256(path)
    manifest = {
        "product": "AutoResearch",
        "product_version": PRODUCT_VERSION,
        "cli_version": CLI_VERSION,
        "api_version": API_VERSION,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "host": host_triple(),
        "files": files,
        "signature": None,
        "signing": "unsigned local artifact; verify manifest checksums and sign in the distribution pipeline",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release": str(output), "files": len(files), "signed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

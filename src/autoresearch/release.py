"""Local-first release platform: configuration, daemon lifecycle, and diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable
from uuid import uuid4

from autoresearch.providers.redaction import redact, redact_text
from autoresearch.versioning import API_VERSION, CLI_VERSION, DATABASE_SCHEMA_VERSION, PRODUCT_VERSION, WORKER_PROTOCOL_VERSION, compatible_api


LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SENSITIVE_CONFIG_KEYS = frozenset({"api_key", "key", "secret", "password", "credential", "authorization"})


@dataclass(frozen=True)
class ProductPaths:
    config_dir: Path
    data_dir: Path
    state_dir: Path
    cache_dir: Path
    install_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "daemon.json"

    @property
    def daemon_log(self) -> Path:
        return self.state_dir / "daemon.log"

    @classmethod
    def current(cls) -> "ProductPaths":
        if sys.platform == "darwin":
            config_default = Path.home() / "Library" / "Application Support" / "AutoResearch"
            data_default = config_default / "data"
            state_default = config_default / "state"
            cache_default = Path.home() / "Library" / "Caches" / "AutoResearch"
            install_default = Path.home() / ".local" / "lib" / "autoresearch"
        elif os.name == "nt":
            appdata = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
            local = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            config_default, data_default = appdata / "AutoResearch", local / "AutoResearch" / "data"
            state_default, cache_default, install_default = local / "AutoResearch" / "state", local / "AutoResearch" / "cache", local / "AutoResearch" / "current"
        else:
            config_default = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "autoresearch"
            data_default = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autoresearch"
            state_default = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "autoresearch"
            cache_default = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "autoresearch"
            install_default = Path.home() / ".local" / "lib" / "autoresearch"
        return cls(
            Path(os.getenv("AUTORESEARCH_CONFIG_HOME", str(config_default))).expanduser(),
            Path(os.getenv("AUTORESEARCH_DATA_HOME", str(data_default))).expanduser(),
            Path(os.getenv("AUTORESEARCH_STATE_HOME", str(state_default))).expanduser(),
            Path(os.getenv("AUTORESEARCH_CACHE_HOME", str(cache_default))).expanduser(),
            Path(os.getenv("AUTORESEARCH_INSTALL_ROOT", str(install_default))).expanduser(),
        )

    def create(self) -> None:
        for path in (self.config_dir, self.data_dir, self.state_dir, self.cache_dir, self.install_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProductConfig:
    backend_url: str = "http://127.0.0.1:8000"
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 8000
    daemon_persistence: str = "preserve"
    provider_config: str | None = None
    allowed_repository_roots: tuple[str, ...] = ()
    tui_executable: str | None = None
    authentication_token_environment_variable: str | None = None
    default_project: str | None = None
    cost_budget_usd: float | None = None
    telemetry_enabled: bool = False
    local_worker_enabled: bool = True

    @property
    def auth_token(self) -> str | None:
        return os.getenv(self.authentication_token_environment_variable or "") or None

    @classmethod
    def load(cls, paths: ProductPaths | None = None) -> "ProductConfig":
        paths = paths or ProductPaths.current()
        if not paths.config_file.exists():
            return cls()
        try:
            return cls.from_mapping(tomllib.loads(paths.config_file.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Configuration is unreadable: {exc}") from exc

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ProductConfig":
        for key, value in raw.items():
            lowered = str(key).lower()
            if (lowered in SENSITIVE_CONFIG_KEYS or "api_key" in lowered or "provider_secret" in lowered) and value not in (None, ""):
                raise ValueError(f"Configuration must not contain raw credential field '{key}'.")
        backend = str(raw.get("backend_url", cls.backend_url)).rstrip("/")
        parsed = urllib.parse.urlparse(backend)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("backend_url must be an HTTP(S) URL without embedded credentials.")
        host = str(raw.get("daemon_host", "127.0.0.1"))
        if host not in LOCAL_HOSTS:
            raise ValueError("daemon_host must be loopback-only; use an authenticated proxy for remote access.")
        port = int(raw.get("daemon_port", parsed.port or 8000))
        if not 1024 <= port <= 65535:
            raise ValueError("daemon_port must be between 1024 and 65535.")
        persistence = str(raw.get("daemon_persistence", "preserve"))
        if persistence not in {"preserve", "stop_on_exit"}:
            raise ValueError("daemon_persistence must be preserve or stop_on_exit.")
        auth_env = raw.get("authentication_token_environment_variable")
        if auth_env is not None:
            auth_env = str(auth_env)
            if not auth_env.isidentifier() or not auth_env.isupper() or any(part in auth_env for part in ("OPENAI", "ANTHROPIC", "GEMINI", "GOOGLE", "UNPAYWALL", "API_KEY")):
                raise ValueError("authentication_token_environment_variable cannot name a provider credential.")
        roots = raw.get("allowed_repository_roots", [])
        if not isinstance(roots, (list, tuple)) or any(not isinstance(item, str) for item in roots):
            raise ValueError("allowed_repository_roots must be a list of path strings.")
        budget = raw.get("cost_budget_usd")
        if budget is not None and float(budget) < 0:
            raise ValueError("cost_budget_usd must be non-negative.")
        optional_path = lambda name: str(Path(str(raw[name])).expanduser()) if raw.get(name) else None
        return cls(
            backend_url=backend, daemon_host=host, daemon_port=port, daemon_persistence=persistence,
            provider_config=optional_path("provider_config"),
            allowed_repository_roots=tuple(str(Path(item).expanduser()) for item in roots),
            tui_executable=optional_path("tui_executable"),
            authentication_token_environment_variable=auth_env,
            default_project=str(raw["default_project"]) if raw.get("default_project") else None,
            cost_budget_usd=float(budget) if budget is not None else None,
            telemetry_enabled=bool(raw.get("telemetry_enabled", False)),
            local_worker_enabled=bool(raw.get("local_worker_enabled", True)),
        )

    def with_updates(self, **updates: Any) -> "ProductConfig":
        values = asdict(self)
        values.update({key: value for key, value in updates.items() if value is not None})
        return ProductConfig.from_mapping(values)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(str(value))


def write_config(paths: ProductPaths, config: ProductConfig) -> Path:
    paths.create()
    values = asdict(config)
    lines = ["# AutoResearch local configuration. Provider keys are never stored here."]
    for key in values:
        value = values[key]
        if value is not None:
            lines.append(f"{key} = {_toml(value)}")
    _atomic_write(paths.config_file, "\n".join(lines) + "\n")
    return paths.config_file


def daemon_environment(config: ProductConfig, paths: ProductPaths, *, repository_root: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"AUTORESEARCH_DATA_ROOT": str(paths.data_dir), "AUTORESEARCH_API_HOST": config.daemon_host, "AUTORESEARCH_API_PORT": str(config.daemon_port)})
    if config.provider_config:
        env["AUTORESEARCH_PROVIDER_CONFIG"] = config.provider_config
    roots = config.allowed_repository_roots or (str((repository_root or Path.cwd()).resolve()),)
    env["AUTORESEARCH_AGENT_ROOTS"] = os.pathsep.join(roots)
    return env


def api_request(config: ProductConfig, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 2.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if config.auth_token:
        headers["Authorization"] = f"Bearer {config.auth_token}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    if body:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(config.backend_url.rstrip("/") + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"backend_http_{exc.code}: {redact_text(exc.read().decode('utf-8', errors='replace'))[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"backend_unreachable: {redact_text(str(exc.reason))}") from exc


def stream_sse(config: ProductConfig, path: str, *, timeout: float = 30.0) -> Iterable[dict[str, Any]]:
    headers = {"Accept": "text/event-stream"}
    if config.auth_token:
        headers["Authorization"] = f"Bearer {config.auth_token}"
    request = urllib.request.Request(config.backend_url.rstrip("/") + path, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            event_type, event_id, payload = "message", None, ""
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if payload:
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            event_type, data = "client.event_error", {"code": "invalid_event_payload"}
                        yield {"id": event_id, "type": event_type, "data": redact(data)}
                    event_type, event_id, payload = "message", None, ""
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("id:"):
                    try:
                        event_id = int(line[3:].strip())
                    except ValueError:
                        event_id = None
                elif line.startswith("data:"):
                    payload += line[5:].strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"backend_unreachable: {redact_text(str(exc.reason))}") from exc


def backend_health(config: ProductConfig) -> dict[str, Any] | None:
    try:
        return api_request(config, "/v1/health")
    except RuntimeError:
        return None


def assert_compatible(health: dict[str, Any]) -> None:
    if health.get("status") != "ready":
        raise RuntimeError(f"backend_unavailable: {health.get('reason') or 'backend did not become ready'}")
    reported = str(health.get("api_version") or health.get("version") or "")
    if not compatible_api(reported):
        raise RuntimeError(f"api_incompatible: CLI requires API {API_VERSION}; backend reported {reported or 'none'}.")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return pid > 0
    except OSError:
        return False


def _is_daemon_process(pid: int) -> bool:
    """Confirm a PID belongs to our daemon before treating or terminating it."""
    if not _alive(pid):
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine"], capture_output=True, text=True, check=False)
        else:
            result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    except OSError:
        return False
    return "autoresearch.release_daemon" in result.stdout


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def start_daemon(config: ProductConfig, paths: ProductPaths, *, timeout_seconds: float = 8.0, repository_root: Path | None = None) -> tuple[dict[str, Any], bool]:
    paths.create()
    current = backend_health(config)
    if current:
        assert_compatible(current)
        return current, False
    if paths.pid_file.exists():
        try:
            pid = int(json.loads(paths.pid_file.read_text(encoding="utf-8")).get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pid = 0
        if _is_daemon_process(pid):
            raise RuntimeError("daemon_starting: local daemon PID exists but health is unavailable; run `autoresearch doctor`.")
        paths.pid_file.unlink(missing_ok=True)
    if not _port_available(config.daemon_host, config.daemon_port):
        raise RuntimeError(f"port_unavailable: {config.daemon_host}:{config.daemon_port} is occupied; change daemon_port or stop the conflicting process.")
    with paths.daemon_log.open("ab") as log:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process = subprocess.Popen([sys.executable, "-m", "autoresearch.release_daemon", "--host", config.daemon_host, "--port", str(config.daemon_port)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt", creationflags=flags, env=daemon_environment(config, paths, repository_root=repository_root))
    _atomic_write(paths.pid_file, json.dumps({"pid": process.pid, "product_version": PRODUCT_VERSION, "api_version": API_VERSION, "started_at": time.time()}) + "\n")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        health = backend_health(config)
        if health:
            assert_compatible(health)
            return health, True
        if process.poll() is not None:
            break
        time.sleep(0.1)
    tail = redact_text(paths.daemon_log.read_text(encoding="utf-8", errors="replace")[-1200:]) if paths.daemon_log.exists() else ""
    raise RuntimeError(f"daemon_start_failed: backend did not become ready. Run `autoresearch doctor`. {tail}")


def stop_daemon(paths: ProductPaths) -> bool:
    if not paths.pid_file.exists():
        return False
    try:
        pid = int(json.loads(paths.pid_file.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        pid = 0
    if _is_daemon_process(pid):
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM)
        else:
            os.killpg(pid, signal.SIGTERM)
    paths.pid_file.unlink(missing_ok=True)
    return True


def resolve_tui(config: ProductConfig, paths: ProductPaths) -> Path | None:
    names = [Path(config.tui_executable).expanduser()] if config.tui_executable else []
    names.append(paths.install_dir / ("autoresearch-tui.exe" if os.name == "nt" else "autoresearch-tui"))
    if located := shutil.which("autoresearch-tui"):
        names.append(Path(located))
    return next((path.resolve() for path in names if path.is_file()), None)


def launch_tui(config: ProductConfig, paths: ProductPaths, *, repository_root: Path | None = None, extra_args: Iterable[str] = ()) -> int:
    _health, started = start_daemon(config, paths, repository_root=repository_root)
    executable = resolve_tui(config, paths)
    if not executable:
        if started and config.daemon_persistence == "stop_on_exit":
            stop_daemon(paths)
        raise RuntimeError("tui_not_installed: no packaged AutoResearch TUI was found. Run the documented installer or configure tui_executable; npx is intentionally not used at runtime.")
    env = dict(os.environ, AUTORESEARCH_BACKEND_URL=config.backend_url)
    result = subprocess.run([str(executable), *extra_args], env=env, check=False)
    if started and config.daemon_persistence == "stop_on_exit":
        stop_daemon(paths)
    return int(result.returncode)


def doctor(config: ProductConfig, paths: ProductPaths, *, check_provider: bool = False) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    add = lambda name, status, detail, remedy="": checks.append({"name": name, "status": status, "detail": redact_text(detail), "remedy": remedy})
    try:
        paths.create(); add("directories", "PASS", "Configuration, data, state, cache, and install directories are writable.")
    except OSError as exc:
        add("directories", "FAIL", str(exc), "Set AUTORESEARCH_*_HOME to writable locations and rerun setup.")
    try:
        ProductConfig.load(paths); add("configuration", "PASS", f"Loaded {paths.config_file}.")
    except ValueError as exc:
        add("configuration", "FAIL", str(exc), "Run `autoresearch setup` to regenerate a safe configuration.")
    add("python", "PASS" if sys.version_info >= (3, 11) else "FAIL", f"Python {sys.version.split()[0]}", "Install Python 3.11 or later.")
    health = backend_health(config)
    add("api_compatibility", "PASS" if health else "WARNING", "Local daemon is compatible." if health else "Local daemon is not running.", "Run `autoresearch` or `autoresearch serve`.")
    add("database_schema", "PASS", f"Expected schema v{DATABASE_SCHEMA_VERSION}; initialized by daemon startup.")
    add("tui", "PASS" if resolve_tui(config, paths) else "WARNING", "Packaged TUI is available." if resolve_tui(config, paths) else "No packaged TUI found.", "Install a local release artifact; runtime intentionally avoids npx.")
    add("git", "PASS" if shutil.which("git") else "WARNING", "Git executable found." if shutil.which("git") else "Git unavailable.", "Install Git for repository inspection.")
    valid_roots = bool(config.allowed_repository_roots) and all(Path(root).exists() for root in config.allowed_repository_roots)
    add("allowed_roots", "PASS" if valid_roots else "WARNING", "Configured roots are usable." if valid_roots else "No persistent allowed repository root is configured.", "Run `autoresearch setup --allowed-root PATH`.")
    if config.provider_config:
        provider = Path(config.provider_config)
        add("provider_configuration", "PASS" if provider.is_file() else "FAIL", f"Provider configuration {'exists' if provider.is_file() else 'is missing'} at {provider}.", "Select a secret-free provider TOML whose key fields name environment variables.")
    else:
        add("provider_configuration", "WARNING", "No provider configuration is selected.", "Run `autoresearch setup --provider-config PATH`; never place provider keys in this file.")
    if not health and not _port_available(config.daemon_host, config.daemon_port):
        add("port", "FAIL", f"{config.daemon_host}:{config.daemon_port} is occupied without healthy AutoResearch.", "Change daemon_port or stop the conflicting process.")
    else:
        add("port", "PASS", f"{config.daemon_host}:{config.daemon_port} is available or owned by AutoResearch.")
    if check_provider:
        add("provider_connectivity", "WARNING", "Live connectivity is intentionally opt-in and belongs to the live smoke matrix.", "Set a budget and run the opt-in live evaluation command.")
    return checks


def _portable(value: str) -> str:
    return value.replace(str(Path.home()), "~") if value else ""


def create_diagnostics(config: ProductConfig, paths: ProductPaths, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("diagnostic output must end with .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    safe = redact(asdict(config))
    for key in ("provider_config", "tui_executable"):
        safe[key] = _portable(str(safe.get(key) or "")) or None
    safe["allowed_repository_roots"] = [_portable(str(root)) for root in config.allowed_repository_roots]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("product.json", json.dumps({"product_version": PRODUCT_VERSION, "cli_version": CLI_VERSION, "api_version": API_VERSION, "database_schema_version": DATABASE_SCHEMA_VERSION, "worker_protocol_version": WORKER_PROTOCOL_VERSION}, indent=2))
        archive.writestr("config.redacted.json", json.dumps(safe, indent=2))
        archive.writestr("doctor.json", json.dumps(doctor(config, paths), indent=2))
        archive.writestr("health.json", json.dumps(backend_health(config) or {"status": "unavailable"}, indent=2))
        if paths.daemon_log.exists():
            archive.writestr("daemon.log", redact_text(paths.daemon_log.read_text(encoding="utf-8", errors="replace")[-100_000:]))
    return output


def verify_release_artifact(source: Path) -> dict[str, Any]:
    import hashlib
    source, manifest_path = source.expanduser().resolve(), source.expanduser().resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("release_manifest_missing: source directory has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("release_manifest_invalid: files map is missing")
    for relative, expected in manifest["files"].items():
        candidate = (source / str(relative)).resolve()
        if source not in candidate.parents or not candidate.is_file():
            raise ValueError(f"release_manifest_invalid: file '{relative}' is missing or escapes release directory")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f"release_checksum_mismatch: '{relative}' did not match manifest")
    return manifest


def stage_update(source: Path, paths: ProductPaths) -> Path:
    manifest = verify_release_artifact(source)
    target = paths.state_dir / "staged" / f"{manifest.get('product_version', 'unknown')}-{uuid4().hex[:8]}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def activate_staged(source: Path, paths: ProductPaths) -> tuple[Path, Path | None]:
    """Activate only the verified TUI payload while retaining a rollback copy."""
    manifest = verify_release_artifact(source)
    tui_name = "autoresearch-tui.exe" if os.name == "nt" else "autoresearch-tui"
    tui_source = source / tui_name
    if not tui_source.is_file():
        raise ValueError("release_tui_missing: activation requires a host TUI binary")
    paths.create()
    previous: Path | None = None
    if paths.install_dir.is_dir() and any(paths.install_dir.iterdir()):
        previous = paths.state_dir / "previous" / f"{manifest.get('product_version', 'unknown')}-{uuid4().hex[:8]}"
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(paths.install_dir, previous)
    paths.install_dir.mkdir(parents=True, exist_ok=True)
    destination = paths.install_dir / tui_name
    shutil.copy2(tui_source, destination)
    if os.name != "nt":
        destination.chmod(destination.stat().st_mode | 0o111)
    return destination, previous


def rollback_update(paths: ProductPaths) -> Path:
    previous_root = paths.state_dir / "previous"
    candidates = sorted((path for path in previous_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True) if previous_root.is_dir() else []
    if not candidates:
        raise ValueError("rollback_unavailable: no retained AutoResearch release exists")
    previous = candidates[0]
    current = paths.install_dir
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(previous, current)
    return current


def remove_installation(paths: ProductPaths, *, purge_data: bool = False) -> list[Path]:
    targets = [paths.install_dir] + ([paths.data_dir, paths.state_dir, paths.cache_dir, paths.config_dir] if purge_data else [])
    removed: list[Path] = []
    for target in targets:
        resolved = target.expanduser().resolve()
        if resolved.name.lower() not in {"autoresearch", "current"}:
            raise ValueError(f"uninstall_refused: unexpected product path {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
            removed.append(resolved)
    return removed

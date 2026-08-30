from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from .backend import MicrotensorBackend
from .binding import write_binding
from .config import ControllerConfig
from .controller import Controller
from .envfile import load_env_file
from .errors import ConfigError
from .redaction import RedactingFilter, redact_text, secret_values
from .state import StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microtensor-miner-controller",
        description="Fail-closed, round-aware supervisor for one Microtensor miner hotkey.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ["MMC_ENV_FILE"]) if os.environ.get("MMC_ENV_FILE") else None,
        help="strict mode-0600 dotenv file (parsed as data, never sourced as shell)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="read-only runtime and registration checks")
    preflight.set_defaults(handler=_preflight)

    run = subparsers.add_parser("run", help="supervise round-specific submission")
    run.add_argument("--once", action="store_true", help="perform one cycle and exit")
    run.set_defaults(handler=_run)
    binding = subparsers.add_parser("bind-selfcheck", help="bind selfcheck to exact artifact")
    binding.set_defaults(handler=_bind_selfcheck)


    status = subparsers.add_parser("status", help="print persisted status JSON")
    _add_state_arguments(status)
    status.set_defaults(handler=_status)

    health = subparsers.add_parser("health", help="print health JSON and fail if stale/unverified")
    _add_state_arguments(health)
    health.add_argument("--max-age-seconds", type=int)
    health.set_defaults(handler=_health)
    return parser


def _add_state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--compact", action="store_true", help="emit JSON on one line")


def _logging(level: str, secrets: tuple[str, ...]) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactingFilter(secrets))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.Formatter.converter = __import__("time").gmtime


def _runtime() -> tuple[ControllerConfig, StateStore, MicrotensorBackend, tuple[str, ...]]:
    secrets = secret_values(os.environ)
    config = ControllerConfig.from_env()
    _logging(config.log_level, secrets)
    state = StateStore(config.state_dir, secrets)
    return config, state, MicrotensorBackend(config), secrets


def _preflight(args: argparse.Namespace) -> int:
    del args
    config, state, backend, _ = _runtime()
    return Controller(config, backend, state).preflight_only()

def _bind_selfcheck(args: argparse.Namespace) -> int:
    del args
    config = ControllerConfig.from_env()
    payload = write_binding(config)
    _print_json(payload, compact=False)
    return 0



def _run(args: argparse.Namespace) -> int:
    config, state, backend, _ = _runtime()
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return Controller(config, backend, state, stop_event=stop).run(once=bool(args.once))


def _state(args: argparse.Namespace) -> StateStore:
    root = args.state_dir
    if root is None:
        root = Path(os.environ.get("MMC_STATE_DIR", "/var/lib/microtensor-miner/controller"))
    return StateStore(Path(root).expanduser().absolute(), secret_values(os.environ))


def _print_json(payload: dict[str, object], compact: bool) -> None:
    if compact:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, sort_keys=True, indent=2))


def _status(args: argparse.Namespace) -> int:
    payload = _state(args).read_status()
    if not payload:
        payload = {"ok": False, "phase": "missing", "message": "no status has been written"}
    _print_json(payload, args.compact)
    return 0 if payload.get("ok") is True else 2


def _health(args: argparse.Namespace) -> int:
    default_age = int(os.environ.get("MMC_HEALTH_MAX_AGE_SECONDS", "180"))
    maximum = args.max_age_seconds if args.max_age_seconds is not None else default_age
    ok, payload = _state(args).health(maximum)
    _print_json(payload, args.compact)
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.env_file is not None:
            load_env_file(args.env_file)
        return int(args.handler(args))
    except ConfigError as exc:
        secrets = secret_values(os.environ)
        print(f"configuration refused: {redact_text(exc, secrets)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

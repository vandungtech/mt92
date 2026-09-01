from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .leaderboard import POLL_SECONDS, RankMonitor
from .state import StateStore


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously record one public Microtensor leaderboard standing."
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=_positive_int, default=POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    monitor_factory: Callable[..., Any] = RankMonitor,
) -> int:
    args = build_parser().parse_args(argv)
    state = StateStore(args.state_dir.expanduser().absolute())
    monitor = monitor_factory(
        state,
        track=args.track,
        hardware_class=args.hardware_class,
        hotkey=args.hotkey,
        poll_seconds=args.poll_seconds,
    )
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = __import__("time").gmtime

    while not stop.is_set():
        payload = monitor.poll_once()
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        if args.once:
            return 0
        stop.wait(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

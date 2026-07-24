#!/usr/bin/env python3
"""Fail-closed process identity checks for the 2026-07-17 salvage queue."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    starttime: int
    argv: tuple[str, ...]


def _read_process(proc_root: Path, pid: int) -> Process:
    process_root = proc_root / str(pid)
    stat_text = (process_root / "stat").read_text(encoding="utf-8")
    close_paren = stat_text.rfind(")")
    if close_paren < 0:
        raise ValueError(f"malformed stat for pid={pid}")
    fields = stat_text[close_paren + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"short stat for pid={pid}")
    cmdline = (process_root / "cmdline").read_bytes()
    argv = tuple(
        token.decode("utf-8", errors="replace")
        for token in cmdline.split(b"\0")
        if token
    )
    return Process(
        pid=pid,
        ppid=int(fields[1]),
        starttime=int(fields[19]),
        argv=argv,
    )


def read_processes(proc_root: Path) -> dict[int, Process]:
    processes: dict[int, Process] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        pid = int(entry.name)
        try:
            process = _read_process(proc_root, pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        processes[pid] = process
    return processes


def descendants(processes: dict[int, Process], root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for process in processes.values():
        children.setdefault(process.ppid, []).append(process.pid)
    found: set[int] = set()
    frontier = list(children.get(root_pid, ()))
    while frontier:
        pid = frontier.pop()
        if pid in found:
            continue
        found.add(pid)
        frontier.extend(children.get(pid, ()))
    return found


def _has_option(argv: tuple[str, ...], option: str, expected: str) -> bool:
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv) and argv[index + 1] == expected:
            return True
        if token == f"{option}={expected}":
            return True
    return False


def is_accelerate_launch(
    process: Process,
    *,
    config: str,
    module: str,
) -> bool:
    argv = process.argv
    accelerate_indexes = [
        index
        for index, token in enumerate(argv)
        if Path(token).name == "accelerate"
    ]
    if not any(
        index + 1 < len(argv) and argv[index + 1] == "launch"
        for index in accelerate_indexes
    ):
        return False
    return _has_option(argv, "--config", config) and _has_option(argv, "-m", module)


def identify_candidate(
    processes: dict[int, Process],
    *,
    root_pid: int,
    root_starttime: int,
    config: str,
    module: str,
) -> tuple[Process | None, str, list[int]]:
    root = processes.get(root_pid)
    if root is None:
        return None, "root_missing", []
    if root.starttime != root_starttime:
        return None, "root_identity_changed", []
    candidate_pids = sorted(
        pid
        for pid in descendants(processes, root_pid)
        if is_accelerate_launch(processes[pid], config=config, module=module)
    )
    if not candidate_pids:
        return None, "not_found", []
    if len(candidate_pids) != 1:
        return None, "ambiguous", candidate_pids
    return processes[candidate_pids[0]], "ready", candidate_pids


def _emit(payload: dict[str, object], exit_code: int) -> int:
    print(json.dumps(payload, sort_keys=True))
    return exit_code


def _identity(args: argparse.Namespace) -> int:
    processes = read_processes(args.proc_root)
    process = processes.get(args.pid)
    if process is None:
        return _emit({"status": "missing", "pid": args.pid}, 3)
    return _emit(
        {
            "status": "ready",
            "pid": process.pid,
            "ppid": process.ppid,
            "starttime": process.starttime,
            "argv": list(process.argv),
        },
        0,
    )


def _identify(args: argparse.Namespace) -> int:
    processes = read_processes(args.proc_root)
    candidate, status, candidate_pids = identify_candidate(
        processes,
        root_pid=args.root_pid,
        root_starttime=args.root_starttime,
        config=args.config,
        module=args.module,
    )
    if candidate is None:
        exit_code = {
            "not_found": 3,
            "ambiguous": 4,
            "root_identity_changed": 5,
            "root_missing": 6,
        }[status]
        return _emit(
            {
                "status": status,
                "root_pid": args.root_pid,
                "root_starttime": args.root_starttime,
                "candidate_pids": candidate_pids,
            },
            exit_code,
        )
    return _emit(
        {
            "status": "ready",
            "root_pid": args.root_pid,
            "root_starttime": args.root_starttime,
            "pid": candidate.pid,
            "starttime": candidate.starttime,
            "argv": list(candidate.argv),
        },
        0,
    )


def _match(args: argparse.Namespace) -> int:
    processes = read_processes(args.proc_root)
    matches = sorted(
        process.pid
        for process in processes.values()
        if (
            is_accelerate_launch(process, config=args.config, module=args.module)
            if args.kind == "accelerate"
            else _has_option(process.argv, "--config", args.config)
            and _has_option(process.argv, "-m", args.module)
        )
    )
    return _emit(
        {
            "status": "ready",
            "kind": args.kind,
            "count": len(matches),
            "pids": matches,
        },
        0,
    )


def _signal(args: argparse.Namespace) -> int:
    processes = read_processes(args.proc_root)
    candidate, status, candidate_pids = identify_candidate(
        processes,
        root_pid=args.root_pid,
        root_starttime=args.root_starttime,
        config=args.config,
        module=args.module,
    )
    if candidate is None:
        return _emit(
            {
                "status": status,
                "candidate_pids": candidate_pids,
                "signal_sent": False,
            },
            4,
        )
    if candidate.pid != args.pid or candidate.starttime != args.starttime:
        return _emit(
            {
                "status": "candidate_identity_changed",
                "expected_pid": args.pid,
                "expected_starttime": args.starttime,
                "actual_pid": candidate.pid,
                "actual_starttime": candidate.starttime,
                "signal_sent": False,
            },
            5,
        )
    payload: dict[str, object] = {
        "status": "would_signal" if args.dry_run else "signaled",
        "pid": candidate.pid,
        "starttime": candidate.starttime,
        "signal": "SIGINT",
        "signal_sent": not args.dry_run,
    }
    if not args.dry_run:
        try:
            os.kill(candidate.pid, signal.SIGINT)
        except ProcessLookupError:
            payload.update(status="process_disappeared", signal_sent=False)
            return _emit(payload, 6)
        except PermissionError:
            payload.update(status="permission_denied", signal_sent=False)
            return _emit(payload, 7)
    return _emit(payload, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity_parser = subparsers.add_parser("identity")
    identity_parser.add_argument("--pid", type=int, required=True)
    identity_parser.set_defaults(func=_identity)

    def add_candidate_arguments(candidate_parser: argparse.ArgumentParser) -> None:
        candidate_parser.add_argument("--root-pid", type=int, required=True)
        candidate_parser.add_argument("--root-starttime", type=int, required=True)
        candidate_parser.add_argument("--config", required=True)
        candidate_parser.add_argument("--module", required=True)

    identify_parser = subparsers.add_parser("identify")
    add_candidate_arguments(identify_parser)
    identify_parser.set_defaults(func=_identify)

    match_parser = subparsers.add_parser("match")
    match_parser.add_argument("--config", required=True)
    match_parser.add_argument("--module", required=True)
    match_parser.add_argument("--kind", choices=("train", "accelerate"), default="train")
    match_parser.set_defaults(func=_match)

    signal_parser = subparsers.add_parser("signal")
    add_candidate_arguments(signal_parser)
    signal_parser.add_argument("--pid", type=int, required=True)
    signal_parser.add_argument("--starttime", type=int, required=True)
    signal_parser.add_argument("--dry-run", action="store_true")
    signal_parser.set_defaults(func=_signal)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

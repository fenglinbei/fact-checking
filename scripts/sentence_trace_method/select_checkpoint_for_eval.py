#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from sft.checkpoint_selection import (
    select_macro_f1_checkpoint,
    select_one_standard_error_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a checkpoint or tau for eval-only checkpoint diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--case-root", required=True)
    checkpoint.add_argument(
        "--policy",
        choices=["current_macro_f1", "current_composite", "macro_f1", "one_standard_error"],
        required=True,
    )
    checkpoint.add_argument("--print-json", action="store_true")

    sync_best = subparsers.add_parser(
        "sync-best",
        description="Materialize a selected checkpoint as a checkpoint alias under <case-root>/train.",
    )
    sync_best.add_argument("--case-root", required=True)
    sync_best.add_argument(
        "--policy",
        choices=["current_macro_f1", "current_composite", "macro_f1", "one_standard_error"],
        default="one_standard_error",
    )
    sync_best.add_argument(
        "--alias",
        default="one_se_best",
        help="Checkpoint alias to create under <case-root>/train. Defaults to one_se_best.",
    )
    sync_best.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    sync_best.add_argument("--force", action="store_true", help="Replace an existing alias.")
    sync_best.add_argument("--dry-run", action="store_true")
    sync_best.add_argument("--selection-path", default=None)
    sync_best.add_argument("--print-json", action="store_true")

    tau = subparsers.add_parser("tau")
    tau.add_argument("--case-root", required=True)
    tau.add_argument("--checkpoint", required=True)
    tau.add_argument("--experiment", required=True)
    tau.add_argument("--taus", required=True)
    tau.add_argument("--metric", default="macro_f1")
    tau.add_argument("--print-json", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--case-root", required=True)
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument(
        "--plan-kind",
        choices=["current_macro_f1", "current_composite", "macro_f1", "one_standard_error"],
        required=True,
    )
    plan.add_argument("--eval-splits", default="val,test")
    plan.add_argument("--tau-grid", default="0,0.25,0.5,0.75,1")
    plan.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "checkpoint":
        selected = _select_checkpoint_by_policy(args.case_root, policy=str(args.policy))
        _print_selected(selected, key="checkpoint", print_json=bool(args.print_json))
        return 0

    if args.command == "sync-best":
        result = _sync_best_alias(
            case_root=Path(args.case_root),
            policy=str(args.policy),
            alias=str(args.alias),
            link_mode=str(args.link_mode),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            selection_path=Path(args.selection_path) if args.selection_path else None,
        )
        if args.print_json:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            _print_sync_result(result)
        return 0

    if args.command == "plan":
        plan = _build_plan(
            case_root=Path(args.case_root),
            checkpoint=str(args.checkpoint),
            plan_kind=str(args.plan_kind),
            eval_splits=_split_list(str(args.eval_splits)),
            tau_grid=[item.strip() for item in str(args.tau_grid).split(",") if item.strip()],
        )
        print(json.dumps(plan, sort_keys=True))
        return 0

    selected_tau = _select_tau(
        case_root=Path(args.case_root),
        checkpoint=str(args.checkpoint),
        experiment=str(args.experiment),
        taus=[item.strip() for item in str(args.taus).split(",") if item.strip()],
        metric=str(args.metric),
    )
    _print_selected(selected_tau, key="tau", print_json=bool(args.print_json))
    return 0


def _select_checkpoint_by_policy(case_root: str | Path, *, policy: str) -> dict[str, object]:
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized in {"current_macro_f1", "current_composite", "macro_f1"}:
        selected = select_macro_f1_checkpoint(case_root)
    elif normalized == "one_standard_error":
        selected = select_one_standard_error_checkpoint(case_root)
    else:
        raise ValueError(f"Unsupported checkpoint selection policy: {policy!r}")
    selected["policy"] = normalized
    return selected


def _sync_best_alias(
    *,
    case_root: Path,
    policy: str,
    alias: str,
    link_mode: str,
    force: bool,
    dry_run: bool,
    selection_path: Path | None,
) -> dict[str, object]:
    _validate_alias(alias)
    selected = _select_checkpoint_by_policy(case_root, policy=policy)
    run_dir = case_root / "train"
    selected_checkpoint = str(selected["checkpoint"])
    source_dir = run_dir / selected_checkpoint
    alias_path = run_dir / alias
    if not source_dir.is_dir():
        raise SystemExit(f"Selected checkpoint directory does not exist: {source_dir}")

    selection_output_path = selection_path or run_dir / f"{alias}_checkpoint_selection.json"
    summary = _selection_summary(
        selected,
        case_root=case_root,
        run_dir=run_dir,
        alias=alias,
        alias_path=alias_path,
        selected_checkpoint_path=source_dir,
        link_mode=link_mode,
        dry_run=dry_run,
        selection_path=selection_output_path,
    )

    if dry_run:
        return summary

    run_dir.mkdir(parents=True, exist_ok=True)
    _materialize_alias(source_dir=source_dir, alias_path=alias_path, link_mode=link_mode, force=force)
    selection_output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _validate_alias(alias: str) -> None:
    alias_path = Path(alias)
    if alias_path.is_absolute() or alias_path.name != alias or alias in {"", ".", ".."}:
        raise SystemExit(f"Alias must be a simple checkpoint directory name, got: {alias!r}")


def _selection_summary(
    selected: dict[str, object],
    *,
    case_root: Path,
    run_dir: Path,
    alias: str,
    alias_path: Path,
    selected_checkpoint_path: Path,
    link_mode: str,
    dry_run: bool,
    selection_path: Path,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "policy": selected["policy"],
        "case_root": str(case_root),
        "run_dir": str(run_dir),
        "alias": alias,
        "alias_path": str(alias_path),
        "selected_checkpoint": str(selected["checkpoint"]),
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "selected_step": int(selected["step"]),
        "selected_macro_f1": float(selected["macro_f1"]),
        "selected_macro_f1_se": float(selected.get("macro_f1_se", 0.0)),
        "link_mode": link_mode,
        "dry_run": bool(dry_run),
        "selection_path": str(selection_path),
    }
    if selected.get("checkpoint_selection_score") is not None:
        summary["selected_checkpoint_selection_score"] = float(selected["checkpoint_selection_score"])
    if selected.get("one_se_best_checkpoint") is not None:
        summary.update(
            {
                "fmax_checkpoint": str(selected["one_se_best_checkpoint"]),
                "fmax_macro_f1": float(selected["one_se_best_macro_f1"]),
                "fmax_macro_f1_se": float(selected["one_se_best_macro_f1_se"]),
                "threshold": float(selected["one_se_threshold"]),
            }
        )
    if selected.get("metrics_path") is not None:
        summary["selected_metrics_path"] = str(selected["metrics_path"])
    return summary


def _materialize_alias(*, source_dir: Path, alias_path: Path, link_mode: str, force: bool) -> None:
    if alias_path.exists() or alias_path.is_symlink():
        if _alias_already_points_to(alias_path, source_dir):
            return
        if not force:
            raise SystemExit(f"Alias already exists and points elsewhere: {alias_path}. Use --force to replace it.")
        _remove_path(alias_path)

    tmp_path = alias_path.with_name(f".{alias_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists() or tmp_path.is_symlink():
        _remove_path(tmp_path)
    if link_mode == "symlink":
        os.symlink(os.path.relpath(source_dir, start=alias_path.parent), tmp_path)
    elif link_mode == "copy":
        shutil.copytree(source_dir, tmp_path)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode!r}")
    tmp_path.rename(alias_path)


def _alias_already_points_to(alias_path: Path, source_dir: Path) -> bool:
    try:
        return alias_path.resolve(strict=True) == source_dir.resolve(strict=True)
    except FileNotFoundError:
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _select_tau(
    *,
    case_root: Path,
    checkpoint: str,
    experiment: str,
    taus: list[str],
    metric: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for tau in taus:
        tag = _tau_tag(tau)
        metrics_path = case_root / "eval" / "val" / checkpoint / f"checkpoint_gap_{experiment}_tau{tag}" / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metric not in metrics:
            continue
        rows.append({"tau": tau, "tag": tag, "metric": float(metrics[metric]), "metrics_path": str(metrics_path)})
    if not rows:
        raise SystemExit(
            f"No val tau metrics found for checkpoint={checkpoint} experiment={experiment} metric={metric}."
        )
    return dict(sorted(rows, key=lambda row: (-float(row["metric"]), float(row["tau"])))[0])


def _build_plan(
    *,
    case_root: Path,
    checkpoint: str,
    plan_kind: str,
    eval_splits: list[str],
    tau_grid: list[str],
) -> list[dict[str, object]]:
    if plan_kind in {"current_macro_f1", "current_composite"}:
        return _fixed_tau_items(
            case_root=case_root,
            checkpoint=checkpoint,
            experiment="E0_current_macro_f1_tau1",
            splits=eval_splits,
            tau="1.0",
        )
    if plan_kind == "macro_f1":
        return [
            *_fixed_tau_items(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E1_val_macro_f1_tau1",
                splits=eval_splits,
                tau="1.0",
            ),
            *_fixed_tau_items(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E2_val_macro_f1_tau0",
                splits=eval_splits,
                tau="0.0",
            ),
            _val_selected_tau_item(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E3_val_macro_f1_val_selected_tau",
                tau_grid=tau_grid,
            ),
        ]
    if plan_kind == "one_standard_error":
        return [
            *_fixed_tau_items(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E4_one_se_tau1",
                splits=eval_splits,
                tau="1.0",
            ),
            *_fixed_tau_items(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E5_one_se_tau0",
                splits=eval_splits,
                tau="0.0",
            ),
            _val_selected_tau_item(
                case_root=case_root,
                checkpoint=checkpoint,
                experiment="E6_one_se_val_selected_tau",
                tau_grid=tau_grid,
            ),
        ]
    raise ValueError(f"Unsupported plan_kind={plan_kind!r}")


def _fixed_tau_items(
    *,
    case_root: Path,
    checkpoint: str,
    experiment: str,
    splits: list[str],
    tau: str,
) -> list[dict[str, object]]:
    return [
        {
            "type": "fixed",
            "experiment": experiment,
            "split": split,
            "logit_adjust": "on",
            "logit_adjust_tau": float(tau),
            "output_dir": str(case_root / "eval" / split / checkpoint / f"checkpoint_gap_{experiment}"),
        }
        for split in splits
    ]


def _val_selected_tau_item(
    *,
    case_root: Path,
    checkpoint: str,
    experiment: str,
    tau_grid: list[str],
) -> dict[str, object]:
    return {
        "type": "val_selected_tau",
        "experiment": experiment,
        "logit_adjust": "on",
        "taus": [float(tau) for tau in tau_grid],
        "metric": "macro_f1",
        "val_output_dir_template": str(
            case_root / "eval" / "val" / checkpoint / f"checkpoint_gap_{experiment}_tau{{tag}}"
        ),
        "test_output_dir_template": str(
            case_root / "eval" / "test" / checkpoint / f"checkpoint_gap_{experiment}_tau{{tag}}"
        ),
    }


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _tau_tag(value: str) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("+", "")


def _print_selected(selected: dict[str, object], *, key: str, print_json: bool) -> None:
    if print_json:
        print(json.dumps(selected, sort_keys=True))
    else:
        print(str(selected[key]))


def _print_sync_result(result: dict[str, object]) -> None:
    print(
        "checkpoint_alias: "
        f"{result['alias']} -> {result['selected_checkpoint']} "
        f"policy={result['policy']} "
        f"macro_f1={float(result['selected_macro_f1']):.6f}"
    )
    if result.get("threshold") is not None:
        print(
            "one_se: "
            f"fmax={result['fmax_checkpoint']} "
            f"fmax_macro_f1={float(result['fmax_macro_f1']):.6f} "
            f"se={float(result['fmax_macro_f1_se']):.6f} "
            f"threshold={float(result['threshold']):.6f}"
        )
    print(f"alias_path: {result['alias_path']}")
    print(f"selection_path: {result['selection_path']}")


if __name__ == "__main__":
    raise SystemExit(main())

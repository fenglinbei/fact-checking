#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        if args.policy in {"current_macro_f1", "current_composite", "macro_f1"}:
            selected = select_macro_f1_checkpoint(args.case_root)
            selected["policy"] = args.policy
        else:
            selected = select_one_standard_error_checkpoint(args.case_root)
            selected["policy"] = args.policy
        _print_selected(selected, key="checkpoint", print_json=bool(args.print_json))
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


if __name__ == "__main__":
    raise SystemExit(main())

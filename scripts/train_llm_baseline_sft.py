from __future__ import annotations

import argparse
import sys

from fact_checking.config import load_yaml
from sft import train_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin entry for SFT training loop.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini-val-size", type=int, default=None)
    parser.add_argument("--mini-val-seed", type=int, default=None)
    parser.add_argument("--prompt-length-stats-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = load_yaml(args.config)

    forwarded = ["--config", args.config]
    if args.mini_val_size is not None:
        forwarded += ["--mini-val-size", str(args.mini_val_size)]
    if args.mini_val_seed is not None:
        forwarded += ["--mini-val-seed", str(args.mini_val_seed)]
    if args.prompt_length_stats_only:
        forwarded.append("--prompt-length-stats-only")

    sys.argv = ["train_loop", *forwarded]
    train_loop.main()


if __name__ == "__main__":
    main()

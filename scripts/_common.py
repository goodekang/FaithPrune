"""Shared argument parsing for the entry-point scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faithprune.config import load_config, save_config  # noqa: E402
from faithprune.utils import setup_logging  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", nargs="+", default=["configs/default.yaml"],
                   help="YAML files merged from left to right")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="overrides such as method.budget=128")
    p.add_argument("--output", default=None, help="output file or directory")
    p.add_argument("--log-level", default="INFO")
    return p


def parse(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, args.overrides)
    return args, cfg


def save_run_config(cfg, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

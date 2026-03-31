from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def run_stub(case_path: Path, config_path: Path, output_dir: Path) -> Path:
    case = _load_yaml(case_path)
    config = _load_yaml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_plan = {
        "status": "stub_only",
        "task": "2d_sd35_flowedit_dnaedit",
        "case_file": str(case_path),
        "config_file": str(config_path),
        "case": case,
        "config": config,
        "next_todo": [
            "Wire real SD3.5 FlowEdit pipeline.",
            "Port DNAEdit conditioning and edit operators.",
            "Replace stub with experiment launcher.",
        ],
    }
    plan_file = output_dir / "run_plan.json"
    plan_file.write_text(json.dumps(run_plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal flowedit-sd3.5 runner stub.")
    parser.add_argument("--case", type=Path, required=True, help="Path to benchmark case yaml.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark config yaml.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for stub results.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan_file = run_stub(args.case, args.config, args.output)
    print(f"[flowedit-sd3.5] Stub run plan generated: {plan_file}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from typing import Any
from pathlib import Path


DEFAULT_ROOT = Path("/")


def copy_dir(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[skip] missing source: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[ok] copied: {src} -> {dst}")
    return True


def resolve_source(source_root: Path, absolute_posix_path: str) -> Path:
    return source_root / Path(absolute_posix_path.lstrip("/"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote json: {path}")


def build_vram_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Record peak VRAM usage per backend/model version for smoke runs.",
        "fields": [
            "version_tag",
            "model_key",
            "model_id",
            "method",
            "precision",
            "gpu_name",
            "peak_vram_gb",
            "measurement_command",
            "measurement_window",
            "notes",
            "timestamp_utc",
        ],
        "entries": [
            {
                "version_tag": "fill-me",
                "model_key": "sd35-large",
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "method": "flowedit|dnaedit",
                "precision": "fp16|bf16|fp32",
                "gpu_name": "fill-me",
                "peak_vram_gb": None,
                "measurement_command": "fill-me",
                "measurement_window": "full-run|inference-only|warmup-excluded",
                "notes": "",
                "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync reusable 2D assets from EditSplat into flowedit-sd3.5."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Filesystem root where '/dev_vepfs/...' is reachable. "
        "Default: '/' (Linux-style absolute paths).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root of flowedit-sd3.5.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional sync manifest JSON output path. Default: docs/upstream_sync_manifest.json.",
    )
    parser.add_argument(
        "--write-vram-template",
        action="store_true",
        help="Write a JSON template for recording per-version peak VRAM usage in future smoke runs.",
    )
    parser.add_argument(
        "--vram-template-out",
        type=Path,
        default=None,
        help="Optional VRAM template output path. Default: docs/smoke_vram_template.json.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    repo_root = args.repo_root.resolve()
    manifest_out = args.manifest_out or (repo_root / "docs" / "upstream_sync_manifest.json")
    vram_template_out = args.vram_template_out or (repo_root / "docs" / "smoke_vram_template.json")

    # 2D-only migration map: no 3D hooks or smoke assets.
    mapping_specs: list[tuple[str, str, str]] = [
        (
            "flowedit_multimodel_src",
            "/dev_vepfs/rc_wu/edit/EditSplat/flowedit_multimodel/src",
            "src/flowedit_sd35/upstream/external/flowedit_multimodel/src",
        ),
        (
            "sandbox_20260309_scripts",
            "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/scripts",
            "benchmarks/2d/upstream_scripts_20260309",
        ),
        (
            "sandbox_20260309_configs",
            "/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/configs",
            "benchmarks/2d/upstream_configs_20260309",
        ),
    ]

    manifest_entries: list[dict[str, Any]] = []
    copied = 0
    for name, src_abs, dst_rel in mapping_specs:
        src = resolve_source(source_root=source_root, absolute_posix_path=src_abs)
        dst = repo_root / dst_rel
        ok = copy_dir(src, dst)
        if ok:
            copied += 1
        manifest_entries.append(
            {
                "name": name,
                "source": str(src),
                "target": str(dst),
                "copied": ok,
            }
        )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "2d_only",
        "copied": copied,
        "total": len(mapping_specs),
        "entries": manifest_entries,
        "vram_tracking_hint": {
            "template_path": str(vram_template_out),
            "note": "Use --write-vram-template to create a per-version VRAM recording template.",
        },
    }
    write_json(path=manifest_out, payload=manifest)

    if args.write_vram_template:
        write_json(path=vram_template_out, payload=build_vram_template())

    print(f"[summary] copied {copied}/{len(mapping_specs)} source directories.")


if __name__ == "__main__":
    main()

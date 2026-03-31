from __future__ import annotations

import argparse
import shutil
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync reusable assets from EditSplat into flowedit-sd3.5 skeleton."
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
    args = parser.parse_args()

    source_root = args.source_root
    repo_root = args.repo_root

    mappings: list[tuple[Path, Path]] = [
        (
            source_root / "dev_vepfs/rc_wu/edit/EditSplat/flowedit_multimodel/src",
            repo_root / "src/upstream/flowedit_multimodel_src",
        ),
        (
            source_root
            / "dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/scripts",
            repo_root / "benchmarks/2d/upstream_scripts_20260309",
        ),
        (
            source_root
            / "dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/configs",
            repo_root / "benchmarks/2d/upstream_configs_20260309",
        ),
        (
            source_root
            / "dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260311_sd35_3d_minimal_grid_diagnose/scripts",
            repo_root / "benchmarks/3d_smoke/upstream_scripts_20260311",
        ),
    ]

    copied = 0
    for src, dst in mappings:
        if copy_dir(src, dst):
            copied += 1
    print(f"[summary] copied {copied}/{len(mappings)} source directories.")


if __name__ == "__main__":
    main()

# Asset Provenance and Migration Map

This document records where reusable assets come from and how they map into this repository.

## Scope

- This repository now tracks **2D-only** upstream migration.
- Any 3D smoke hooks, 3D benchmarks, and 3D diagnostic assets are intentionally excluded.

## 2D Upstream Sources

1. `/dev_vepfs/rc_wu/edit/EditSplat/flowedit_multimodel/src/`
2. `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/scripts`
3. `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/configs`

## Target Mapping in `flowedit-sd3.5`

| Upstream source | Target path in this repo | Status |
| --- | --- | --- |
| `flowedit_multimodel/src/` | `src/flowedit_sd35/upstream/external/flowedit_multimodel/src/` | pending copy |
| `...20260309.../scripts` | `benchmarks/2d/upstream_scripts_20260309/` | pending copy |
| `...20260309.../configs` | `benchmarks/2d/upstream_configs_20260309/` | pending copy |

## Current Session Note

- In this session, the `/dev_vepfs/...` paths are not directly accessible from the current Windows shell context.
- Therefore v1 uses templates and stubs first.
- Use `scripts/sync_assets_from_editsplat.py` on a machine/context where these paths are visible.

## Sync Script Outputs

`scripts/sync_assets_from_editsplat.py` now writes:

1. `docs/upstream_sync_manifest.json` (default path): copy result manifest.
2. Optional VRAM template via `--write-vram-template`:
   - default path: `docs/smoke_vram_template.json`
   - purpose: reserve a reusable schema for per-version VRAM usage recording.

## VRAM Recording Contract (for future smoke)

Track peak VRAM usage per backend/model version with these fields:

- `version_tag` (commit/model bundle identifier)
- `model_key`, `model_id`, `method`
- `precision`, `gpu_name`
- `peak_vram_gb`
- `measurement_command`, `measurement_window`
- `timestamp_utc`, `notes`

Recommended command pattern (example):

```bash
python scripts/run_2d_stub.py ...  # plus your runtime VRAM probe wrapper
```

Keep one record per version update so memory regression can be compared across runs.

## Migration Principles (2D v1)

- Keep this repo standalone; do not depend on EditSplat runtime path layout.
- Import incrementally: start from 2D SD3.5 FlowEdit + DNAEdit critical path.
- Backend bridge priority: `flowedit_adapters.py` -> `core_backend.py` -> `model_registry.py`.

# Asset Provenance and Migration Map

This document records where reusable assets come from and how they map into this repository.

## Upstream Sources

1. `/dev_vepfs/rc_wu/edit/EditSplat/flowedit_multimodel/src/`
2. `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/scripts`
3. `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/configs`
4. `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260311_sd35_3d_minimal_grid_diagnose/scripts/`

## Target Mapping in `flowedit-sd3.5`

| Upstream source | Target path in this repo | Status |
| --- | --- | --- |
| `flowedit_multimodel/src/` | `src/upstream/flowedit_multimodel_src/` | pending copy |
| `...20260309.../scripts` | `benchmarks/2d/upstream_scripts_20260309/` | pending copy |
| `...20260309.../configs` | `benchmarks/2d/upstream_configs_20260309/` | pending copy |
| `...20260311.../scripts` | `benchmarks/3d_smoke/upstream_scripts_20260311/` | pending copy |

## Current Session Note

- In this session, the `/dev_vepfs/...` paths are not directly accessible from the current Windows shell context.
- Therefore v1 uses templates and stubs first.
- Use `scripts/sync_assets_from_editsplat.py` on a machine/context where these paths are visible.

## Migration Principles for v1

- Keep this repo standalone; do not depend on EditSplat runtime path layout.
- Import incrementally: start from 2D SD3.5 FlowEdit + DNAEdit critical path.
- Keep 3D as smoke-level compatibility hooks until 2D path is stable.

# flowedit-sd3.5 (v1 skeleton)

`flowedit-sd3.5` is a standalone repository skeleton for:
- 2D SD3.5 FlowEdit + DNAEdit (primary target)
- 3D smoke benchmark hooks (minimal diagnostic path)

This first version focuses on structure and runnable stubs, not full model migration.

## Repository Layout

```text
src/flowedit_sd35/          # package skeleton + minimal runner logic
benchmarks/2d/              # 2D cases/config templates
benchmarks/3d_smoke/        # 3D smoke cases/config templates
docs/                       # asset provenance + bootstrap notes
scripts/                    # local run and asset sync scripts
```

## Source Asset Provenance (to be migrated incrementally)

Current reusable assets are planned from:
- `/dev_vepfs/rc_wu/edit/EditSplat/flowedit_multimodel/src/`
- `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/scripts`
- `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/configs`
- `/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260311_sd35_3d_minimal_grid_diagnose/scripts/`

See [docs/ASSET_PROVENANCE.md](./docs/ASSET_PROVENANCE.md) for mapping and status.

## Quick Start (stub)

```bash
python scripts/run_2d_stub.py ^
  --case benchmarks/2d/cases/dnaedit_sd35_minimal.yaml ^
  --config benchmarks/2d/configs/flowedit_sd35_base.yaml ^
  --output runs/2d_stub
```

The command writes `run_plan.json` into `--output` and validates that config templates can be loaded.

## Notes

- This repo intentionally does not bulk-copy the full EditSplat implementation in v1.
- Use `scripts/sync_assets_from_editsplat.py` when source paths are accessible on the current machine.

# flowedit-sd3.5 (2D bootstrap)

`flowedit-sd3.5` is a standalone skeleton focused on the minimal real 2D path for:
- SD3.5 FlowEdit
- DNAEdit

This stage keeps the workflow lightweight and locally verifiable. It validates manifests/configs and runner entry contracts without requiring model launch.

## Repository Layout

```text
src/flowedit_sd35/          # package skeleton + runner entrypoint
benchmarks/2d/              # pure 2D cases/config templates
docs/                       # bootstrap notes
scripts/                    # local run and asset sync scripts
tests/                      # lightweight contract/path tests
```

## Minimal Real 2D Path

```bash
python scripts/run_2d_stub.py ^
  --case benchmarks/2d/cases/dnaedit_sd35_real_minimal.yaml ^
  --config benchmarks/2d/configs/flowedit_sd35_real_minimal.yaml ^
  --output runs/2d_real_minimal_contract
```

The command writes `run_plan.json` and is used as a local contract check for case/config wiring.

## Validation Requirements

- Every version must keep at least one 2D smoke case/config pair runnable at manifest/path level.
- Every version smoke must record 显存用量 (VRAM usage).
- Required smoke VRAM metrics: `peak_vram_mb`, `allocated_vram_mb`, `reserved_vram_mb`.
- The benchmark config must explicitly declare this requirement under `validation.smoke`.

## Local Verification

```bash
python -m unittest -v tests.test_minimal_real_2d_contract
```

## Latest Smoke

Latest real 2D smoke measurements are tracked in [docs/SMOKE_RESULTS_20260401.md](./docs/SMOKE_RESULTS_20260401.md).

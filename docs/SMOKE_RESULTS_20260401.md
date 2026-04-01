# 2D Smoke Results (2026-04-01)

Environment:
- host: `dev-intern-02`
- python: `/dev_vepfs/rc_wu/envs/editsplat_multimodel_v2/bin/python`
- repo clone: `/dev_vepfs/rc_wu/_codex_staging/20260401_flowedit_sd35_validate`
- source image: `/dev_vepfs/rc_wu/_codex_staging/20260331_editsplat_A_experiment/dataset/dataset/face/_images_/frame_00001.jpg`
- GPU binding: `CUDA_VISIBLE_DEVICES=3`

Results:

| version_tag | method | model_key | status | peak_vram_mb | reserved_vram_mb | runtime_sec | output_image |
|---|---|---|---|---:|---:|---:|---|
| `flowedit_sd35_base` | `flowedit` | `sd35-medium-turbo-open` | `success` | `16620.55` | `17236.00` | `28.30` | `/dev_vepfs/rc_wu/_codex_staging/20260401_flowedit_sd35_validate/runs/flowedit_face_clown_smoke_v2/outputs/face_clown_flowedit_smoke.png` |
| `dnaedit_sd35_base` | `dnaedit` | `sd35-medium-turbo-open` | `success` | `18732.70` | `19350.00` | `30.71` | `/dev_vepfs/rc_wu/_codex_staging/20260401_flowedit_sd35_validate/runs/dnaedit_face_clown_smoke_v1/outputs/face_clown_dnaedit_smoke.png` |

Raw manifests:
- `/dev_vepfs/rc_wu/_codex_staging/20260401_flowedit_sd35_validate/runs/flowedit_face_clown_smoke_v2/manifest.json`
- `/dev_vepfs/rc_wu/_codex_staging/20260401_flowedit_sd35_validate/runs/dnaedit_face_clown_smoke_v1/manifest.json`

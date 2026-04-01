from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

RUNNER_VERSION = "0.2.0"
RUNNER_TASK = "2d_sd35_flowedit_dnaedit"
METRIC_PREFIX = "FLOWEDIT_SD35_METRIC"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _import_backend():
    errors: list[str] = []
    try:
        from flowedit_sd35.upstream import FlowBackendConfig, FlowEditCoreBackend
        return FlowBackendConfig, FlowEditCoreBackend, None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"flowedit_sd35.upstream: {type(exc).__name__}: {exc}")
    try:
        from flowedit_multimodel.src.core_backend import FlowBackendConfig, FlowEditCoreBackend
        return FlowBackendConfig, FlowEditCoreBackend, None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"flowedit_multimodel.src.core_backend: {type(exc).__name__}: {exc}")
    return None, None, RuntimeError(" | ".join(errors))


def _tensor_from_pil(image: Image.Image):
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _pil_from_tensor(image_t):
    import torch
    from PIL import Image

    x = image_t[0].detach().float().cpu()
    if x.min().item() < 0.0:
        x = (x + 1.0) * 0.5
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _build_stub_manifest(
    case_path: Path,
    config_path: Path,
    output_dir: Path,
    case: dict[str, Any],
    config: dict[str, Any],
    reason: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = Path(str(case.get("outputs", {}).get("edited_image", "edited.png")))
    if requested.is_absolute():
        requested = Path(requested.name)
    output_image_path = (output_dir / requested).resolve()
    return _write_json(
        output_dir / "manifest.json",
        {
            "runner_version": RUNNER_VERSION,
            "task": RUNNER_TASK,
            "status": "stub_unavailable",
            "reason": reason,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_file": str(case_path),
            "config_file": str(config_path),
            "output_image": str(output_image_path),
            "smoke_metrics": {
                "metric_prefix": METRIC_PREFIX,
                "parse_keys": [
                    "version_tag",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                    "edit_latency_sec",
                ],
            },
        },
    )


def run_stub(case_path: Path, config_path: Path, output_dir: Path) -> Path:
    return run_case(case_path=case_path, config_path=config_path, output_dir=output_dir)


def run_case(case_path: Path, config_path: Path, output_dir: Path) -> Path:
    case = _load_yaml(case_path)
    config = _load_yaml(config_path)
    FlowBackendConfig, FlowEditCoreBackend, import_error = _import_backend()
    if import_error is not None:
        return _build_stub_manifest(
            case_path=case_path,
            config_path=config_path,
            output_dir=output_dir,
            case=case,
            config=config,
            reason=f"upstream backend unavailable: {type(import_error).__name__}: {import_error}",
        )

    try:
        import torch
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return _build_stub_manifest(
            case_path=case_path,
            config_path=config_path,
            output_dir=output_dir,
            case=case,
            config=config,
            reason=f"runtime dependency unavailable: {type(exc).__name__}: {exc}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    case_inputs = case.get("inputs", {})
    case_prompts = case.get("prompts", {})
    model_cfg = config.get("model", {})
    pipeline_cfg = config.get("pipeline", {})
    dna_cfg = config.get("dnaedit", {})

    source_image_raw = str(case_inputs.get("source_image", "")).strip()
    if not source_image_raw:
        return _write_json(
            output_dir / "manifest.json",
            {
                "runner_version": RUNNER_VERSION,
                "task": RUNNER_TASK,
                "status": "input_error",
                "reason": "Missing case.inputs.source_image",
                "case_file": str(case_path),
                "config_file": str(config_path),
            },
        )
    source_image_path = _resolve_path(source_image_raw, case_path.parent)
    if not source_image_path.exists():
        return _write_json(
            output_dir / "manifest.json",
            {
                "runner_version": RUNNER_VERSION,
                "task": RUNNER_TASK,
                "status": "input_error",
                "reason": f"source image not found: {source_image_path}",
                "case_file": str(case_path),
                "config_file": str(config_path),
            },
        )

    source_prompt = str(case_prompts.get("source", case_inputs.get("source_prompt", ""))).strip()
    target_prompt = str(
        case_prompts.get(
            "target",
            case_inputs.get("target_prompt", case_inputs.get("dna_prompt", case.get("description", ""))),
        )
    ).strip()
    negative_prompt = str(case_prompts.get("negative", "")).strip()
    output_name = Path(str(case.get("outputs", {}).get("edited_image", "edited.png")))
    if output_name.is_absolute():
        output_name = Path(output_name.name)
    output_image_path = (output_dir / output_name).resolve()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    config_obj = FlowBackendConfig(
        model_key=str(model_cfg.get("key", "sd35-large")).strip(),
        model_id=str(model_cfg.get("model_id", "")).strip(),
        method=str(pipeline_cfg.get("edit_method", "flowedit")).strip(),
        hf_home=str(model_cfg.get("hf_home", "")).strip(),
        adapter_resize_side=int(model_cfg.get("adapter_resize_side", 512)),
        adapter_gpu=int(model_cfg.get("adapter_gpu", -1)),
        hf_token=str(model_cfg.get("hf_token", "")).strip(),
        dna_steps=int(dna_cfg.get("steps", 40)),
        dna_src_guidance_scale=float(dna_cfg.get("src_guidance_scale", 1.0)),
        dna_tar_guidance_scale=float(dna_cfg.get("tar_guidance_scale", 3.5)),
        dna_t_start=int(dna_cfg.get("t_start", 13)),
        dna_mvg=float(dna_cfg.get("mvg", 0.8)),
    )

    dna_runtime_root = str(dna_cfg.get("runtime_root", "")).strip()
    if dna_runtime_root:
        import os

        os.environ["EDITSPLAT_DNAEDIT_RUNTIME_ROOT"] = dna_runtime_root

    try:
        image_pil = Image.open(source_image_path).convert("RGB")
        image_t = _tensor_from_pil(image_pil)

        runtime_t0 = time.time()
        backend = FlowEditCoreBackend(
            config=config_obj,
            project_root=str(Path(__file__).resolve().parents[2]),
        )
        device_name = str(backend.device)
        peak_allocated = 0
        peak_reserved = 0
        if backend.device.type == "cuda":
            device_name = torch.cuda.get_device_name(backend.device)
            torch.cuda.reset_peak_memory_stats(backend.device)

        edited_t = backend.edit(
            image=image_t,
            src_prompt=source_prompt,
            tar_prompt=target_prompt,
            negative_prompt=negative_prompt,
            diffusion_steps=int(pipeline_cfg.get("steps", 24)),
            n_avg=int(pipeline_cfg.get("n_avg", 1)),
            src_guidance_scale=float(pipeline_cfg.get("src_guidance_scale", 2.5)),
            tar_guidance_scale=float(pipeline_cfg.get("tar_guidance_scale", 9.0)),
            n_min=int(pipeline_cfg.get("n_min", 0)),
            n_max=int(pipeline_cfg.get("n_max", 14)),
            seed=int(case_inputs.get("seed", 0)),
        )
        edit_latency_sec = float(time.time() - runtime_t0)

        if backend.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(backend.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(backend.device))

        _pil_from_tensor(edited_t).save(output_image_path)
        version_tag = str(config.get("experiment", {}).get("name", model_cfg.get("family", config_obj.model_key)))
        manifest = {
            "runner_version": RUNNER_VERSION,
            "task": RUNNER_TASK,
            "status": "success",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_file": str(case_path),
            "config_file": str(config_path),
            "source_image": str(source_image_path),
            "output_image": str(output_image_path),
            "method": config_obj.method,
            "model_key": config_obj.model_key,
            "runtime_sec": edit_latency_sec,
            "backend_summary": backend.summarize(),
            "smoke_metrics": {
                "metric_prefix": METRIC_PREFIX,
                "parse_keys": [
                    "version_tag",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                    "edit_latency_sec",
                ],
                "version_tag": version_tag,
                "gpu_peak_allocated_bytes": peak_allocated,
                "gpu_peak_reserved_bytes": peak_reserved,
                "edit_latency_sec": edit_latency_sec,
            },
            "vram": {
                "device": device_name,
                "adapter_gpu": int(config_obj.adapter_gpu),
                "peak_vram_mb": peak_allocated / (1024 * 1024),
                "allocated_vram_mb": peak_allocated / (1024 * 1024),
                "reserved_vram_mb": peak_reserved / (1024 * 1024),
            },
        }
        print(
            f"{METRIC_PREFIX} "
            f"version_tag={version_tag} "
            f"gpu_peak_allocated_bytes={peak_allocated} "
            f"gpu_peak_reserved_bytes={peak_reserved} "
            f"edit_latency_sec={edit_latency_sec:.6f}"
        )
        return _write_json(output_dir / "manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001
        return _write_json(
            output_dir / "manifest.json",
            {
                "runner_version": RUNNER_VERSION,
                "task": RUNNER_TASK,
                "status": "runtime_error",
                "reason": f"{type(exc).__name__}: {exc}",
                "case_file": str(case_path),
                "config_file": str(config_path),
                "source_image": str(source_image_path),
                "output_image": str(output_image_path),
                "smoke_metrics": {
                    "metric_prefix": METRIC_PREFIX,
                    "parse_keys": [
                        "version_tag",
                        "gpu_peak_allocated_bytes",
                        "gpu_peak_reserved_bytes",
                        "edit_latency_sec",
                    ],
                },
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2D single-case flowedit-sd3.5 runner (real backend if available, otherwise stub_unavailable)."
    )
    parser.add_argument("--case", type=Path, required=True, help="Path to 2D case yaml.")
    parser.add_argument("--config", type=Path, required=True, help="Path to 2D config yaml.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for image + manifest.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = run_case(args.case, args.config, args.output)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"[flowedit-sd3.5] status={payload.get('status')} manifest={manifest_path}")
        if payload.get("status") == "success":
            print(f"[flowedit-sd3.5] output_image={payload.get('output_image')}")
        elif payload.get("reason"):
            print(f"[flowedit-sd3.5] reason={payload.get('reason')}")
    except Exception:  # noqa: BLE001
        print(f"[flowedit-sd3.5] Result written: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from .flowedit_adapters import FlowEditParams, create_adapter


FLOWEDIT_COMPAT_MODEL_KEYS = (
    "flux1-dev",
    "flux2-dev",
    "flux2-klein-4b",
    "sd35-large",
    "sd35-large-turbo",
    "sd35-medium-turbo-open",
    "sd3-medium",
    "z-image",
)

DNAEDIT_COMPAT_PREFIXES = ("flux1", "sd3", "sd35")


@dataclass
class FlowBackendConfig:
    model_key: str = "sd35-large"
    model_id: str = ""
    method: str = "flowedit"
    hf_home: str = "/dev-vepfs/rc_wu/rc_wu/cache/hf_home"
    adapter_resize_side: int = 512
    adapter_gpu: int = -1
    hf_token: str = ""
    dna_steps: int = 40
    dna_src_guidance_scale: float = 1.0
    dna_tar_guidance_scale: float = 3.5
    dna_t_start: int = 13
    dna_mvg: float = 0.8


def _tensor_to_pil01(img: torch.Tensor) -> Image.Image:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    x = img[0].detach().float()
    if x.min() < 0.0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def _crop_to_16(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    return image.crop((0, 0, w - (w % 16), h - (h % 16)))


def _pick_device(config: FlowBackendConfig) -> torch.device:
    if torch.cuda.is_available():
        if int(config.adapter_gpu) >= 0:
            return torch.device(f"cuda:{int(config.adapter_gpu)}")
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_dtype(model_key: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float16
    if model_key.startswith("flux2"):
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _ensure_ipdb_stub() -> None:
    if "ipdb" in sys.modules:
        return
    stub = types.ModuleType("ipdb")
    stub.set_trace = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["ipdb"] = stub


def _load_dnaedit_utils(runtime_root: Path):
    _ensure_ipdb_stub()
    module_path = runtime_root / "scripts" / "DNAEdit_utils.py"
    spec = importlib.util.spec_from_file_location("dnaedit_utils_core_backend", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DNAEdit_FLUX, module.DNAEdit_SD3


def _resolve_dnaedit_runtime_root(project_root: Path) -> Path:
    candidates = []
    env_root = os.environ.get("EDITSPLAT_DNAEDIT_RUNTIME_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            project_root
            / "sandboxes"
            / "20260309_rfedit_dnaedit_flowalign_snredit"
            / "runtime"
            / "DNAEdit_code_http11",
            Path("/dev_vepfs/rc_wu/edit/EditSplat/sandboxes/20260309_rfedit_dnaedit_flowalign_snredit/runtime/DNAEdit_code_http11"),
        ]
    )
    for candidate in candidates:
        if (candidate / "scripts" / "DNAEdit_utils.py").is_file():
            return candidate
    searched = [str(x) for x in candidates]
    raise FileNotFoundError(
        "DNAEdit runtime not found. searched=" + "; ".join(searched)
    )


class FlowEditCoreBackend:
    def __init__(self, config: FlowBackendConfig, project_root: str):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.device = _pick_device(config)
        self.dtype = _pick_dtype(config.model_key, self.device)
        if config.method == "dnaedit":
            self.dtype = torch.float16
        self.runtime_records: List[Dict[str, float]] = []

        if config.model_key not in FLOWEDIT_COMPAT_MODEL_KEYS:
            raise ValueError(
                f"Unsupported FlowEdit-compatible model_key={config.model_key}. "
                f"Supported: {', '.join(FLOWEDIT_COMPAT_MODEL_KEYS)}"
            )

        os.environ["HF_HOME"] = config.hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(config.hf_home) / "hub")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        self.adapter = create_adapter(
            model_key=config.model_key,
            device=self.device,
            dtype=self.dtype,
            hf_token=config.hf_token or os.environ.get("HF_TOKEN", ""),
            cache_dir=str(Path(config.hf_home) / "hub"),
            override_model_id=config.model_id or None,
        )
        self.pipe = self.adapter.pipe
        self._dna_flux = None
        self._dna_sd3 = None

    def _ensure_dnaedit(self) -> None:
        if self._dna_flux is not None and self._dna_sd3 is not None:
            return
        if not self.config.model_key.startswith(DNAEDIT_COMPAT_PREFIXES):
            raise ValueError(f"DNAEdit is not supported for model_key={self.config.model_key}")
        runtime_root = _resolve_dnaedit_runtime_root(self.project_root)
        self._dna_flux, self._dna_sd3 = _load_dnaedit_utils(runtime_root)

    def _encode_image(self, image_pil: Image.Image) -> torch.Tensor:
        image_pil = _crop_to_16(image_pil)
        image_src = self.pipe.image_processor.preprocess(image_pil)
        image_src = image_src.to(self.device, dtype=self.pipe.vae.dtype)
        ctx = torch.autocast(device_type="cuda") if self.device.type == "cuda" else torch.inference_mode()
        with ctx, torch.inference_mode():
            x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
        return (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

    def _run_dnaedit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str) -> torch.Tensor:
        self._ensure_dnaedit()
        x0_src = self._encode_image(image_pil)
        method_key = self.config.model_key
        if method_key.startswith("flux1"):
            x0_tar = self._dna_flux(
                self.pipe,
                self.pipe.scheduler,
                x0_src,
                src_prompt,
                tar_prompt,
                "",
                int(self.config.dna_steps),
                float(self.config.dna_src_guidance_scale),
                float(self.config.dna_tar_guidance_scale),
                int(self.config.dna_t_start),
                float(self.config.dna_mvg),
            )
        else:
            x0_tar = self._dna_sd3(
                self.pipe,
                self.pipe.scheduler,
                x0_src,
                src_prompt,
                tar_prompt,
                "",
                int(self.config.dna_steps),
                float(self.config.dna_src_guidance_scale),
                float(self.config.dna_tar_guidance_scale),
                int(self.config.dna_t_start),
                float(self.config.dna_mvg),
            )
        x0_tar_denorm = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        ctx = torch.autocast(device_type="cuda") if self.device.type == "cuda" else torch.inference_mode()
        with ctx, torch.inference_mode():
            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        return image_tar.to(torch.float32)

    def edit(
        self,
        image: torch.Tensor,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: str,
        diffusion_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
        seed: int,
    ) -> torch.Tensor:
        image_pil = _tensor_to_pil01(image)
        t0 = time.time()
        if self.config.method == "flowedit":
            params = FlowEditParams(
                diffusion_steps=int(diffusion_steps),
                n_avg=int(n_avg),
                src_guidance_scale=float(src_guidance_scale),
                tar_guidance_scale=float(tar_guidance_scale),
                n_min=int(n_min),
                n_max=int(n_max),
                seed=int(seed),
                negative_prompt=str(negative_prompt or ""),
                resize_side=int(self.config.adapter_resize_side),
            )
            out = self.adapter.edit(image_pil, src_prompt, tar_prompt, params)
        elif self.config.method == "dnaedit":
            out = self._run_dnaedit(image_pil, src_prompt, tar_prompt)
        else:
            raise NotImplementedError(
                f"External backend currently supports flowedit/dnaedit only, got method={self.config.method}"
            )

        self.runtime_records.append(
            {
                "runtime_sec": float(time.time() - t0),
            }
        )
        return out.to(image.device, dtype=torch.float32)

    def summarize(self) -> Dict[str, float]:
        if not self.runtime_records:
            return {"count": 0, "avg_runtime_sec": 0.0}
        runtimes = [x["runtime_sec"] for x in self.runtime_records]
        return {
            "count": float(len(runtimes)),
            "avg_runtime_sec": float(sum(runtimes) / len(runtimes)),
            "max_runtime_sec": float(max(runtimes)),
            "min_runtime_sec": float(min(runtimes)),
        }



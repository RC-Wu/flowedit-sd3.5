from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .flowedit_adapters import FlowEditParams, create_adapter
from .model_registry import get_model_spec


@dataclass
class FlowBackendConfig:
    model_key: str = "sd35-large"
    model_id: str = ""
    method: str = "flowedit"
    hf_home: str = ""
    adapter_resize_side: int = 512
    adapter_gpu: int = -1
    hf_token: str = ""
    dna_steps: int = 40
    dna_src_guidance_scale: float = 1.0
    dna_tar_guidance_scale: float = 3.5
    dna_t_start: int = 13
    dna_mvg: float = 0.8
    dtype: str = "float16"

    def normalized_method(self) -> str:
        return str(self.method).strip().lower()


def _resolve_device(adapter_gpu: int) -> Any:
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return f"cuda:{adapter_gpu}" if adapter_gpu >= 0 else "cpu"

    if adapter_gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{adapter_gpu}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_name: str, device: Any) -> Any:
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return dtype_name or "float16"

    name = str(dtype_name).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name in mapping:
        return mapping[name]
    return torch.float16 if getattr(device, "type", "cpu") == "cuda" else torch.float32


def _resolve_cache_dir(config: FlowBackendConfig, project_root: Path) -> Path:
    if config.hf_home.strip():
        return Path(config.hf_home).expanduser().resolve() / "hub"
    return project_root / ".cache" / "huggingface" / "hub"


def _to_pil_image(image: Any) -> Any:
    from PIL import Image  # type: ignore

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        tensor = image.detach()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor of shape [B,C,H,W] or [C,H,W], got {tuple(image.shape)}")

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] == 4:
            tensor = tensor[:3]
        elif tensor.shape[0] != 3:
            raise ValueError(f"Unsupported tensor channel count: {tensor.shape[0]}")

        tensor = tensor.float().cpu()
        min_v = float(tensor.min().item())
        max_v = float(tensor.max().item())
        if min_v >= -1.0 and max_v <= 1.0 and min_v < 0.0:
            tensor = (tensor + 1.0) / 2.0
        elif min_v >= 0.0 and max_v <= 1.0:
            pass
        else:
            tensor = tensor.clamp(0.0, 255.0) / 255.0

        array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        return Image.fromarray(array, mode="RGB")

    raise TypeError("Unsupported image type. Expected PIL.Image, path, or torch.Tensor.")


class FlowEditCoreBackend:
    """Minimal backend bridge that delegates to synced upstream SD3.5 adapters."""

    def __init__(self, config: FlowBackendConfig, project_root: Optional[str] = None):
        self.config = config
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[4]
        )
        self.device = _resolve_device(config.adapter_gpu)
        self.dtype = _resolve_dtype(config.dtype, self.device)
        self.cache_dir = _resolve_cache_dir(config=config, project_root=self.project_root)
        self._configure_hf_env()

        self.adapter = create_adapter(
            model_key=config.model_key,
            device=self.device,
            dtype=self.dtype,
            hf_token=config.hf_token.strip() or None,
            cache_dir=str(self.cache_dir),
            override_model_id=config.model_id.strip() or None,
            project_root=self.project_root,
        )

    def _configure_hf_env(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hf_home = str(self.cache_dir.parent)
        if self.config.hf_home.strip():
            hf_home = str(Path(self.config.hf_home).expanduser())
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", str(self.cache_dir))

    def _resolve_params(
        self,
        diffusion_steps: Optional[int],
        n_avg: int,
        src_guidance_scale: Optional[float],
        tar_guidance_scale: Optional[float],
        n_min: Optional[int],
        n_max: Optional[int],
        seed: int,
        negative_prompt: str,
    ) -> FlowEditParams:
        spec = get_model_spec(self.config.model_key)
        method = self.config.normalized_method()

        if method == "dnaedit":
            resolved_steps = int(diffusion_steps if diffusion_steps is not None else self.config.dna_steps)
            resolved_src_g = float(
                src_guidance_scale if src_guidance_scale is not None else self.config.dna_src_guidance_scale
            )
            resolved_tar_g = float(
                tar_guidance_scale if tar_guidance_scale is not None else self.config.dna_tar_guidance_scale
            )
            resolved_n_min = int(n_min if n_min is not None else max(0, self.config.dna_t_start))
        else:
            resolved_steps = int(diffusion_steps if diffusion_steps is not None else spec.default_steps)
            resolved_src_g = float(
                src_guidance_scale if src_guidance_scale is not None else spec.default_src_guidance
            )
            resolved_tar_g = float(
                tar_guidance_scale if tar_guidance_scale is not None else spec.default_tar_guidance
            )
            resolved_n_min = int(n_min if n_min is not None else spec.default_n_min)

        resolved_n_max = int(n_max if n_max is not None else spec.default_n_max)
        return FlowEditParams(
            diffusion_steps=resolved_steps,
            n_avg=int(n_avg),
            src_guidance_scale=resolved_src_g,
            tar_guidance_scale=resolved_tar_g,
            n_min=resolved_n_min,
            n_max=resolved_n_max,
            seed=int(seed),
            negative_prompt=negative_prompt,
            resize_side=int(self.config.adapter_resize_side),
        )

    def edit(
        self,
        image: Any,
        src_prompt: str,
        tar_prompt: str,
        negative_prompt: str = "",
        diffusion_steps: Optional[int] = None,
        n_avg: int = 1,
        src_guidance_scale: Optional[float] = None,
        tar_guidance_scale: Optional[float] = None,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        seed: int = 0,
    ) -> Any:
        image_pil = _to_pil_image(image)
        params = self._resolve_params(
            diffusion_steps=diffusion_steps,
            n_avg=n_avg,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_min=n_min,
            n_max=n_max,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        return self.adapter.edit(
            image=image_pil,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            params=params,
        )

    @property
    def adapter_available(self) -> bool:
        return bool(getattr(self.adapter, "available", False))


from __future__ import annotations

import importlib
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .model_registry import ModelSpec, get_model_spec


@dataclass
class FlowEditParams:
    diffusion_steps: int
    n_avg: int
    src_guidance_scale: float
    tar_guidance_scale: float
    n_min: int
    n_max: int
    seed: int = 0
    negative_prompt: str = ""
    resize_side: int = 512


class BaseAdapter(ABC):
    @abstractmethod
    def edit(
        self,
        image: Any,
        src_prompt: str,
        tar_prompt: str,
        params: FlowEditParams,
    ) -> Any:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class MissingAdapter(BaseAdapter):
    def __init__(self, spec: ModelSpec, reason: str):
        self.spec = spec
        self.reason = reason

    def edit(
        self,
        image: Any,
        src_prompt: str,
        tar_prompt: str,
        params: FlowEditParams,
    ) -> Any:
        del image, src_prompt, tar_prompt, params
        raise RuntimeError(
            "FlowEdit adapter is unavailable. "
            f"model_key={self.spec.key}, model_id={self.spec.model_id}, reason={self.reason}"
        )

    @property
    def available(self) -> bool:
        return False


class UpstreamDelegatingAdapter(BaseAdapter):
    def __init__(self, upstream_adapter: Any, upstream_module: Any):
        self._upstream_adapter = upstream_adapter
        self._upstream_module = upstream_module

    def edit(
        self,
        image: Any,
        src_prompt: str,
        tar_prompt: str,
        params: FlowEditParams,
    ) -> Any:
        upstream_params_cls = getattr(self._upstream_module, "FlowEditParams", None)
        if upstream_params_cls is None:
            raise RuntimeError("Upstream flowedit_adapters.py is missing FlowEditParams.")

        upstream_params = upstream_params_cls(
            diffusion_steps=params.diffusion_steps,
            n_avg=params.n_avg,
            src_guidance_scale=params.src_guidance_scale,
            tar_guidance_scale=params.tar_guidance_scale,
            n_min=params.n_min,
            n_max=params.n_max,
            seed=params.seed,
            negative_prompt=params.negative_prompt,
            resize_side=params.resize_side,
        )
        return self._upstream_adapter.edit(
            image_pil=image,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            params=upstream_params,
        )


def _iter_upstream_roots(project_root: Path) -> list[Path]:
    return [
        project_root / "src" / "flowedit_sd35" / "upstream" / "external" / "flowedit_multimodel",
        project_root / "src" / "upstream" / "flowedit_multimodel",
    ]


def _load_upstream_module(project_root: Path) -> Optional[Any]:
    module_name = "src.flowedit_adapters"
    for root in _iter_upstream_roots(project_root):
        src_file = root / "src" / "flowedit_adapters.py"
        if not src_file.is_file():
            continue
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        cached_module = sys.modules.get(module_name)
        cached_file = Path(getattr(cached_module, "__file__", "")).resolve() if cached_module else None
        if cached_file is not None and cached_file == src_file.resolve():
            module = cached_module
        else:
            sys.modules.pop(module_name, None)
            sys.modules.pop("src.model_registry", None)
            module = importlib.import_module(module_name)
        if hasattr(module, "create_adapter"):
            return module
    return None


def create_adapter(
    model_key: str,
    device: Any,
    dtype: Any,
    hf_token: Optional[str],
    cache_dir: str,
    override_model_id: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> BaseAdapter:
    spec = get_model_spec(model_key)
    if spec.loader != "sd3":
        return MissingAdapter(spec=spec, reason="This repository currently wires only sd3 loaders (2D-only).")

    if project_root is None:
        project_root = Path(__file__).resolve().parents[4]
    try:
        upstream_module = _load_upstream_module(project_root=project_root)
    except Exception as exc:  # noqa: BLE001
        return MissingAdapter(spec=spec, reason=f"Failed to import upstream flowedit_adapters.py: {exc}")

    if upstream_module is None:
        return MissingAdapter(
            spec=spec,
            reason=(
                "Upstream flowedit_multimodel sources are not synced. "
                "Run scripts/sync_assets_from_editsplat.py first."
            ),
        )

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return MissingAdapter(spec=spec, reason=f"PyTorch import failed: {exc}")

    resolved_device = device
    if resolved_device is None:
        gpu = int(os.environ.get("FLOWEDIT_ADAPTER_GPU", "-1"))
        if gpu >= 0 and torch.cuda.is_available():
            resolved_device = torch.device(f"cuda:{gpu}")
        else:
            resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = torch.float16 if getattr(resolved_device, "type", "cpu") == "cuda" else torch.float32

    model_id = override_model_id if override_model_id else spec.model_id
    adapter = upstream_module.create_adapter(
        model_key=model_key,
        device=resolved_device,
        dtype=resolved_dtype,
        hf_token=hf_token,
        cache_dir=cache_dir,
        override_model_id=model_id,
    )
    return UpstreamDelegatingAdapter(upstream_adapter=adapter, upstream_module=upstream_module)

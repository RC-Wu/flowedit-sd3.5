from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    loader: str
    gated: bool
    note: str
    default_steps: int
    default_src_guidance: float
    default_tar_guidance: float
    default_n_min: int
    default_n_max: int


# 2D-only registry for this repository.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "sd35-large": ModelSpec(
        key="sd35-large",
        model_id="stabilityai/stable-diffusion-3.5-large",
        loader="sd3",
        gated=True,
        note="Official SD3.5 large release.",
        default_steps=32,
        default_src_guidance=3.5,
        default_tar_guidance=10.5,
        default_n_min=0,
        default_n_max=20,
    ),
    "sd35-large-turbo": ModelSpec(
        key="sd35-large-turbo",
        model_id="stabilityai/stable-diffusion-3.5-large-turbo",
        loader="sd3",
        gated=True,
        note="Official SD3.5 large turbo release.",
        default_steps=24,
        default_src_guidance=2.5,
        default_tar_guidance=8.0,
        default_n_min=0,
        default_n_max=16,
    ),
    "sd35-medium-turbo-open": ModelSpec(
        key="sd35-medium-turbo-open",
        model_id=os.environ.get(
            "FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID",
            "tensorart/stable-diffusion-3.5-medium-turbo",
        ),
        loader="sd3",
        gated=False,
        note="Open fallback when gated official SD3.5 repos are unavailable.",
        default_steps=24,
        default_src_guidance=2.5,
        default_tar_guidance=8.0,
        default_n_min=0,
        default_n_max=16,
    ),
}


def list_model_specs() -> Dict[str, ModelSpec]:
    return dict(MODEL_SPECS)


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model key: {key}. Available: {available}") from exc


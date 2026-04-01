from dataclasses import dataclass
from typing import Dict

import os


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


MODEL_SPECS: Dict[str, ModelSpec] = {
    # Baseline from existing EditSplat code.
    "flux1-dev": ModelSpec(
        key="flux1-dev",
        model_id="black-forest-labs/FLUX.1-dev",
        loader="flux1",
        gated=True,
        note="Current EditSplat baseline.",
        default_steps=28,
        default_src_guidance=1.5,
        default_tar_guidance=6.5,
        default_n_min=0,
        default_n_max=24,
    ),
    # Official latest FLUX.2 (gated).
    "flux2-dev": ModelSpec(
        key="flux2-dev",
        model_id="black-forest-labs/FLUX.2-dev",
        loader="flux2",
        gated=True,
        note="Official latest FLUX.2 image-editing pipeline.",
        default_steps=28,
        default_src_guidance=1.5,
        default_tar_guidance=5.0,
        default_n_min=0,
        default_n_max=24,
    ),
    # Open FLUX.2 variant for immediate runnable experiments.
    "flux2-klein-4b": ModelSpec(
        key="flux2-klein-4b",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        loader="flux2-klein",
        gated=False,
        note="Open FLUX.2 Klein 4B variant. Best with Python>=3.10 and diffusers main for guidance_embeds compatibility.",
        default_steps=24,
        default_src_guidance=1.2,
        default_tar_guidance=4.5,
        default_n_min=0,
        default_n_max=20,
    ),
    # Official latest SD models (gated).
    "sd35-large": ModelSpec(
        key="sd35-large",
        model_id="stabilityai/stable-diffusion-3.5-large",
        loader="sd3",
        gated=True,
        note="Official latest SD3.5 large.",
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
        note="Official latest SD3.5 turbo.",
        default_steps=24,
        default_src_guidance=2.5,
        default_tar_guidance=8.0,
        default_n_min=0,
        default_n_max=16,
    ),
    # Open SD3.5 fallback (community diffusers repo) for environments without Stability gating access.
    "sd35-medium-turbo-open": ModelSpec(
        key="sd35-medium-turbo-open",
        model_id=os.environ.get(
            "FLOWEDIT_SD35_MEDIUM_TURBO_OPEN_MODEL_ID",
            "tensorart/stable-diffusion-3.5-medium-turbo",
        ),
        loader="sd3",
        gated=False,
        note="Open SD3.5 medium turbo fallback when official Stability SD3.5 repos are gated.",
        default_steps=24,
        default_src_guidance=2.5,
        default_tar_guidance=8.0,
        default_n_min=0,
        default_n_max=16,
    ),
    # Qwen official image editing model.
    "qwen-image-edit": ModelSpec(
        key="qwen-image-edit",
        model_id="Qwen/Qwen-Image-Edit",
        loader="qwen-image-edit",
        gated=False,
        note="Qwen image-edit model with explicit source-image conditioning branch.",
        default_steps=28,
        default_src_guidance=2.0,
        default_tar_guidance=5.5,
        default_n_min=0,
        default_n_max=22,
    ),
    # Z-Image is text-to-image only; we test FlowEdit latent editing compatibility.
    "z-image": ModelSpec(
        key="z-image",
        model_id="Tongyi-MAI/Z-Image",
        loader="z-image",
        gated=False,
        note="Flow-matching T2I model; retained for compatibility evaluation on latent-space FlowEdit.",
        default_steps=28,
        default_src_guidance=2.0,
        default_tar_guidance=5.0,
        default_n_min=0,
        default_n_max=20,
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {key}. Available: {', '.join(MODEL_SPECS.keys())}")
    return MODEL_SPECS[key]


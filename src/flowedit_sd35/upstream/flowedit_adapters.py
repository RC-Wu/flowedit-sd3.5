from __future__ import annotations

import inspect
import json
import math
import os
import random
import shutil
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
try:
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
except ImportError:
    Qwen2TokenizerFast = None
    Qwen3ForCausalLM = None

try:
    from diffusers import AuraFlowPipeline
except ImportError:
    AuraFlowPipeline = None

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    QwenImageEditPipeline = None

try:
    from diffusers import ZImagePipeline
except ImportError:
    ZImagePipeline = None

try:
    from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
except ImportError:
    AutoencoderKLFlux2 = None
    Flux2Transformer2DModel = None

try:
    from .flux2_klein_pipeline_local import Flux2KleinPipeline, compute_empirical_mu, retrieve_timesteps as retrieve_timesteps_flux2
except ImportError:
    Flux2KleinPipeline = None
    compute_empirical_mu = None
    retrieve_timesteps_flux2 = None
from .model_registry import get_model_spec


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pil_to_tensor01(image: Image.Image, side: Optional[int] = None) -> torch.Tensor:
    if side is not None and image.size != (side, side):
        image = image.resize((side, side), Image.BILINEAR)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


def tensor_m11_to_pil(img_bchw: torch.Tensor) -> Image.Image:
    x = img_bchw[0].detach().float().cpu()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def scale_noise(
    scheduler,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


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


def _run_flowedit_loop(
    x_src: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    params: FlowEditParams,
    velocity_src: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    velocity_tar: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    zt_edit = x_src.clone()
    xt_tar = None
    diffusion_steps = len(timesteps)

    for i, t in enumerate(timesteps):
        if diffusion_steps - i > params.n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if diffusion_steps - i > params.n_min:
            v_delta_avg = torch.zeros_like(x_src)
            for _ in range(params.n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1.0 - t_i) * x_src + t_i * fwd_noise
                zt_tar_in = zt_edit + zt_src - x_src

                v_src = velocity_src(zt_src, t)
                v_tar = velocity_tar(zt_tar_in, t)
                v_delta_avg = v_delta_avg + (v_tar - v_src) / float(params.n_avg)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg.to(torch.float32)
            zt_edit = zt_edit.to(v_delta_avg.dtype)
        else:
            if i == diffusion_steps - params.n_min:
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            v_tar = velocity_tar(xt_tar, t)
            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (t_im1 - t_i) * v_tar.to(torch.float32)
            xt_tar = xt_tar.to(v_tar.dtype)

    return zt_edit if params.n_min == 0 else xt_tar


class BaseAdapter:
    def __init__(self, pipe, device: torch.device):
        self.pipe = pipe
        self.device = device
        self.model_dtype = next(self.pipe.transformer.parameters()).dtype

    def edit(
        self,
        image_pil: Image.Image,
        src_prompt: str,
        tar_prompt: str,
        params: FlowEditParams,
    ) -> torch.Tensor:
        raise NotImplementedError


class Flux1Adapter(BaseAdapter):
    def _encode_src(self, image_pil: Image.Image, side: int):
        image_t = _pil_to_tensor01(image_pil, side=side).to(self.device)
        img_pil = Image.fromarray((image_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        w, h = img_pil.size
        wc, hc = w - (w % 16), h - (h % 16)
        if (wc, hc) != (w, h):
            img_pil = img_pil.crop((0, 0, wc, hc))

        image_src = self.pipe.image_processor.preprocess(img_pil).to(self.device, dtype=self.pipe.vae.dtype)
        shift = float(getattr(self.pipe.vae.config, "shift_factor", 0.0))
        scale = float(getattr(self.pipe.vae.config, "scaling_factor", 1.0))
        with torch.autocast("cuda"), torch.inference_mode():
            x_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
        x_lat = (x_denorm - shift) * scale

        # diffusers>=0.35: prepare/unpack use pixel-space height/width.
        orig_h = x_lat.shape[2] * self.pipe.vae_scale_factor
        orig_w = x_lat.shape[3] * self.pipe.vae_scale_factor
        nc = self.pipe.transformer.config.in_channels // 4
        x_lat, img_ids = self.pipe.prepare_latents(
            batch_size=x_lat.shape[0],
            num_channels_latents=nc,
            height=orig_h,
            width=orig_w,
            dtype=x_lat.dtype,
            device=x_lat.device,
            generator=None,
            latents=x_lat,
        )
        x_packed = self.pipe._pack_latents(x_lat, x_lat.shape[0], nc, x_lat.shape[2], x_lat.shape[3])
        return x_packed, img_ids, orig_h, orig_w, shift, scale

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)

        x_src, latent_img_ids, orig_h, orig_w, shift, scale = self._encode_src(image_pil, params.resize_side)

        scheduler = self.pipe.scheduler
        mu = calculate_shift(
            x_src.shape[1],
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1.0 / params.diffusion_steps, params.diffusion_steps)
        timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, self.device, sigmas=sigmas, mu=mu)

        src_emb, src_pool, src_txt_ids = self.pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=self.device)
        tar_emb, tar_pool, tar_txt_ids = self.pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, device=self.device)

        src_emb = src_emb.to(self.model_dtype)
        src_pool = src_pool.to(self.model_dtype)
        tar_emb = tar_emb.to(self.model_dtype)
        tar_pool = tar_pool.to(self.model_dtype)
        x_src = x_src.to(self.model_dtype)

        if self.pipe.transformer.config.guidance_embeds:
            src_guid = torch.tensor([params.src_guidance_scale], device=self.device, dtype=self.model_dtype).expand(x_src.shape[0])
            tar_guid = torch.tensor([params.tar_guidance_scale], device=self.device, dtype=self.model_dtype).expand(x_src.shape[0])
        else:
            src_guid = None
            tar_guid = None

        def _velocity(latents, t, prompt_embeds, pooled, txt_ids, guidance):
            tt = t.expand(latents.shape[0]).to(self.model_dtype)
            return self.pipe.transformer(
                hidden_states=latents,
                timestep=tt / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=latent_img_ids,
                pooled_projections=pooled,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        out_packed = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(z, t, src_emb, src_pool, src_txt_ids, src_guid),
            velocity_tar=lambda z, t: _velocity(z, t, tar_emb, tar_pool, tar_txt_ids, tar_guid),
        )

        out_lat = self.pipe._unpack_latents(out_packed, orig_h, orig_w, self.pipe.vae_scale_factor)
        out_lat = (out_lat / scale) + shift
        image = self.pipe.vae.decode(out_lat.to(self.pipe.vae.dtype), return_dict=False)[0]
        return image


class Flux2KleinAdapter(BaseAdapter):
    def _encode_src(self, image_pil: Image.Image, side: int):
        image = image_pil.resize((side, side), Image.BILINEAR)
        w, h = image.size
        multiple_of = self.pipe.vae_scale_factor * 2
        w = (w // multiple_of) * multiple_of
        h = (h // multiple_of) * multiple_of
        image = self.pipe.image_processor.preprocess(image, height=h, width=w, resize_mode="crop")
        image = image.to(self.device, dtype=self.pipe.vae.dtype)
        x_img = self.pipe._encode_vae_image(image=image, generator=None)
        img_ids = self.pipe._prepare_latent_ids(x_img).to(self.device)
        x_src = self.pipe._pack_latents(x_img).to(self.model_dtype)
        return x_src, img_ids

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)

        x_src, latent_img_ids = self._encode_src(image_pil, params.resize_side)
        scheduler = self.pipe.scheduler

        sigmas = np.linspace(1.0, 1.0 / params.diffusion_steps, params.diffusion_steps)
        mu = compute_empirical_mu(image_seq_len=x_src.shape[1], num_steps=params.diffusion_steps)
        timesteps, _ = retrieve_timesteps_flux2(scheduler, params.diffusion_steps, self.device, sigmas=sigmas, mu=mu)

        src_emb, src_txt_ids = self.pipe.encode_prompt(
            prompt=src_prompt, device=self.device, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
        )
        tar_emb, tar_txt_ids = self.pipe.encode_prompt(
            prompt=tar_prompt, device=self.device, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
        )
        src_neg_emb, src_neg_txt_ids = self.pipe.encode_prompt(
            prompt="", device=self.device, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
        )
        tar_neg_emb, tar_neg_txt_ids = self.pipe.encode_prompt(
            prompt="", device=self.device, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
        )

        src_emb = src_emb.to(self.model_dtype)
        tar_emb = tar_emb.to(self.model_dtype)
        src_neg_emb = src_neg_emb.to(self.model_dtype)
        tar_neg_emb = tar_neg_emb.to(self.model_dtype)

        def _velocity(latents, t, cond_emb, cond_ids, neg_emb, neg_ids, guidance_scale):
            tt = t.expand(latents.shape[0]).to(self.model_dtype)
            # Some Flux2 transformer builds (diffusers main) support guidance_embeds=False
            # correctly and should receive real guidance scales; older builds do not and can
            # collapse if fed non-zero guidance embeddings from randomly initialized branches.
            try:
                sig = inspect.signature(self.pipe.transformer.__class__.__init__)
                has_guidance_embeds_arg = "guidance_embeds" in sig.parameters
            except Exception:  # noqa: BLE001
                has_guidance_embeds_arg = False

            guidance_value = float(guidance_scale) if has_guidance_embeds_arg else 0.0
            guidance = torch.full((latents.shape[0],), guidance_value, device=latents.device, dtype=torch.float32)
            cond = self.pipe.transformer(
                hidden_states=latents,
                timestep=tt / 1000,
                guidance=guidance,
                encoder_hidden_states=cond_emb,
                txt_ids=cond_ids,
                img_ids=latent_img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            cond = cond[:, : latents.shape[1], :]

            # Distilled FLUX.2 models do not need explicit negative branch CFG.
            if getattr(getattr(self.pipe, "config", None), "is_distilled", False):
                return cond

            if guidance_scale <= 1.0:
                return cond

            neg = self.pipe.transformer(
                hidden_states=latents,
                timestep=tt / 1000,
                guidance=guidance,
                encoder_hidden_states=neg_emb,
                txt_ids=neg_ids,
                img_ids=latent_img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            neg = neg[:, : latents.shape[1], :]
            return neg + guidance_scale * (cond - neg)

        out_packed = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(z, t, src_emb, src_txt_ids, src_neg_emb, src_neg_txt_ids, params.src_guidance_scale),
            velocity_tar=lambda z, t: _velocity(z, t, tar_emb, tar_txt_ids, tar_neg_emb, tar_neg_txt_ids, params.tar_guidance_scale),
        )

        latents = self.pipe._unpack_latents_with_ids(out_packed, latent_img_ids)
        bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * bn_std + bn_mean
        latents = self.pipe._unpatchify_latents(latents)
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        return image


class SD3Adapter(BaseAdapter):
    def _encode_src(self, image_pil: Image.Image, side: int):
        image_t = _pil_to_tensor01(image_pil, side=side).to(self.device)
        image_t = image_t.to(dtype=self.pipe.vae.dtype)
        with torch.autocast("cuda"), torch.inference_mode():
            x_denorm = self.pipe.vae.encode(image_t * 2 - 1).latent_dist.mode()
        shift = float(getattr(self.pipe.vae.config, "shift_factor", 0.0))
        scale = float(getattr(self.pipe.vae.config, "scaling_factor", 1.0))
        x_src = (x_denorm - shift) * scale
        return x_src.to(self.model_dtype), shift, scale

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)
        x_src, shift, scale = self._encode_src(image_pil, params.resize_side)
        scheduler = self.pipe.scheduler
        timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, self.device)

        src = self.pipe.encode_prompt(
            prompt=src_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=params.negative_prompt if params.negative_prompt else "",
            do_classifier_free_guidance=True,
            device=self.device,
        )
        tar = self.pipe.encode_prompt(
            prompt=tar_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=params.negative_prompt if params.negative_prompt else "",
            do_classifier_free_guidance=True,
            device=self.device,
        )
        src_pos, src_neg, src_pool_pos, src_pool_neg = [x.to(self.model_dtype) for x in src]
        tar_pos, tar_neg, tar_pool_pos, tar_pool_neg = [x.to(self.model_dtype) for x in tar]

        def _velocity(latents, t, pos_emb, neg_emb, pos_pool, neg_pool, guidance):
            latent_in = torch.cat([latents, latents], dim=0)
            timestep = t.expand(latent_in.shape[0])
            enc = torch.cat([neg_emb, pos_emb], dim=0)
            pool = torch.cat([neg_pool, pos_pool], dim=0)
            pred = self.pipe.transformer(
                hidden_states=latent_in,
                timestep=timestep,
                encoder_hidden_states=enc,
                pooled_projections=pool,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            p_u, p_c = pred.chunk(2)
            return p_u + guidance * (p_c - p_u)

        out_lat = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(z, t, src_pos, src_neg, src_pool_pos, src_pool_neg, params.src_guidance_scale),
            velocity_tar=lambda z, t: _velocity(z, t, tar_pos, tar_neg, tar_pool_pos, tar_pool_neg, params.tar_guidance_scale),
        )

        out_denorm = (out_lat / scale) + shift
        image = self.pipe.vae.decode(out_denorm.to(self.pipe.vae.dtype), return_dict=False)[0]
        return image


class QwenImageEditAdapter(BaseAdapter):
    def _prepare_src_image(self, image_pil: Image.Image, side: int):
        image = image_pil.resize((side, side), Image.BILINEAR)
        w, h = image.size
        multiple_of = self.pipe.vae_scale_factor * 2
        w = (w // multiple_of) * multiple_of
        h = (h // multiple_of) * multiple_of
        image = self.pipe.image_processor.resize(image, h, w)
        image_t = self.pipe.image_processor.preprocess(image, h, w).unsqueeze(2).to(self.device, dtype=self.pipe.vae.dtype)
        return image, image_t, h, w

    def _encode_src(self, image_t: torch.Tensor):
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        x_img = self.pipe._encode_vae_image(image=image_t, generator=None)
        latent_h, latent_w = x_img.shape[3], x_img.shape[4]
        x_src = self.pipe._pack_latents(
            x_img,
            x_img.shape[0],
            num_channels_latents,
            latent_h,
            latent_w,
        ).to(self.model_dtype)
        return x_src

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)

        prompt_image, image_t, h, w = self._prepare_src_image(image_pil, params.resize_side)
        x_src = self._encode_src(image_t=image_t)
        image_latents = x_src.clone()

        scheduler = self.pipe.scheduler
        sigmas = np.linspace(1.0, 1.0 / params.diffusion_steps, params.diffusion_steps)
        mu = calculate_shift(
            x_src.shape[1],
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, self.device, sigmas=sigmas, mu=mu)

        negative_prompt = params.negative_prompt if params.negative_prompt else ""
        src_emb, src_mask = self.pipe.encode_prompt(
            image=prompt_image,
            prompt=src_prompt,
            device=self.device,
            max_sequence_length=512,
        )
        tar_emb, tar_mask = self.pipe.encode_prompt(
            image=prompt_image,
            prompt=tar_prompt,
            device=self.device,
            max_sequence_length=512,
        )
        src_neg_emb, src_neg_mask = self.pipe.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            device=self.device,
            max_sequence_length=512,
        )
        tar_neg_emb, tar_neg_mask = self.pipe.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            device=self.device,
            max_sequence_length=512,
        )

        src_emb = src_emb.to(self.model_dtype)
        tar_emb = tar_emb.to(self.model_dtype)
        src_neg_emb = src_neg_emb.to(self.model_dtype)
        tar_neg_emb = tar_neg_emb.to(self.model_dtype)

        src_txt_seq_lens = src_mask.sum(dim=1).tolist() if src_mask is not None else None
        tar_txt_seq_lens = tar_mask.sum(dim=1).tolist() if tar_mask is not None else None
        src_neg_txt_seq_lens = src_neg_mask.sum(dim=1).tolist() if src_neg_mask is not None else None
        tar_neg_txt_seq_lens = tar_neg_mask.sum(dim=1).tolist() if tar_neg_mask is not None else None

        im_h = h // self.pipe.vae_scale_factor // 2
        im_w = w // self.pipe.vae_scale_factor // 2
        img_shapes = [[(1, im_h, im_w), (1, im_h, im_w)]] * x_src.shape[0]

        def _velocity(
            latents,
            t,
            cond_emb,
            cond_mask,
            cond_txt_lens,
            neg_emb,
            neg_mask,
            neg_txt_lens,
            true_cfg_scale,
            distilled_guidance_scale,
        ):
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            guidance = None
            if self.pipe.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [latents.shape[0]],
                    float(distilled_guidance_scale),
                    device=latents.device,
                    dtype=torch.float32,
                )

            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=cond_mask,
                encoder_hidden_states=cond_emb,
                img_shapes=img_shapes,
                txt_seq_lens=cond_txt_lens,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if true_cfg_scale <= 1.0:
                return noise_pred

            neg_noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=neg_mask,
                encoder_hidden_states=neg_emb,
                img_shapes=img_shapes,
                txt_seq_lens=neg_txt_lens,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).clamp_min(1e-6)
            return comb_pred * (cond_norm / noise_norm)

        out_packed = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(
                z,
                t,
                src_emb,
                src_mask,
                src_txt_seq_lens,
                src_neg_emb,
                src_neg_mask,
                src_neg_txt_seq_lens,
                params.src_guidance_scale,
                params.src_guidance_scale,
            ),
            velocity_tar=lambda z, t: _velocity(
                z,
                t,
                tar_emb,
                tar_mask,
                tar_txt_seq_lens,
                tar_neg_emb,
                tar_neg_mask,
                tar_neg_txt_seq_lens,
                params.tar_guidance_scale,
                params.tar_guidance_scale,
            ),
        )

        out_lat = self.pipe._unpack_latents(out_packed, h, w, self.pipe.vae_scale_factor).to(self.pipe.vae.dtype)
        z_dim = int(getattr(self.pipe.vae.config, "z_dim", out_lat.shape[2]))
        latents_mean = torch.tensor(self.pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(out_lat.device, out_lat.dtype)
        latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(out_lat.device, out_lat.dtype)
        out_lat = out_lat / latents_std + latents_mean
        image = self.pipe.vae.decode(out_lat, return_dict=False)[0][:, :, 0]
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        return image


class ZImageAdapter(BaseAdapter):
    def _encode_src(self, image_pil: Image.Image, side: int):
        image_t = _pil_to_tensor01(image_pil, side=side).to(self.device)
        image_t = image_t.to(dtype=self.pipe.vae.dtype)
        with torch.autocast("cuda"), torch.inference_mode():
            x_denorm = self.pipe.vae.encode(image_t * 2 - 1).latent_dist.mode()
        shift = float(getattr(self.pipe.vae.config, "shift_factor", 0.0))
        scale = float(getattr(self.pipe.vae.config, "scaling_factor", 1.0))
        x_src = (x_denorm - shift) * scale
        return x_src.to(torch.float32), shift, scale

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)
        x_src, shift, scale = self._encode_src(image_pil, params.resize_side)
        scheduler = self.pipe.scheduler

        image_seq_len = (x_src.shape[2] // 2) * (x_src.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.sigma_min = 0.0
        sigmas = np.linspace(1.0, 1.0 / params.diffusion_steps, params.diffusion_steps)
        timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, self.device, sigmas=sigmas, mu=mu)

        negative_prompt = params.negative_prompt if params.negative_prompt else ""
        src_pos, src_neg = self.pipe.encode_prompt(
            prompt=src_prompt,
            device=self.device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            max_sequence_length=512,
        )
        tar_pos, tar_neg = self.pipe.encode_prompt(
            prompt=tar_prompt,
            device=self.device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            max_sequence_length=512,
        )

        def _to_device_list(xs: List[torch.Tensor]) -> List[torch.Tensor]:
            return [x.to(self.device) for x in xs]

        src_pos = _to_device_list(src_pos)
        src_neg = _to_device_list(src_neg)
        tar_pos = _to_device_list(tar_pos)
        tar_neg = _to_device_list(tar_neg)

        def _velocity(latents, t, pos_embeds, neg_embeds, guidance):
            timestep = t.expand(latents.shape[0]).to(torch.float32)
            timestep = (1000.0 - timestep) / 1000.0

            latent_input = latents.to(self.pipe.transformer.dtype).unsqueeze(2)
            latent_input_list = list(latent_input.unbind(dim=0))

            pos_out = self.pipe.transformer(
                latent_input_list,
                timestep,
                pos_embeds,
                return_dict=False,
            )[0]

            if guidance <= 1.0:
                noise_pred = torch.stack([x.float() for x in pos_out], dim=0)
                return -noise_pred.squeeze(2)

            neg_out = self.pipe.transformer(
                latent_input_list,
                timestep,
                neg_embeds,
                return_dict=False,
            )[0]

            guided = []
            for p, n in zip(pos_out, neg_out):
                p_f = p.float()
                n_f = n.float()
                guided.append(p_f + guidance * (p_f - n_f))
            noise_pred = torch.stack(guided, dim=0)
            return -noise_pred.squeeze(2)

        out_lat = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(z, t, src_pos, src_neg, params.src_guidance_scale),
            velocity_tar=lambda z, t: _velocity(z, t, tar_pos, tar_neg, params.tar_guidance_scale),
        )

        out_lat = torch.nan_to_num(out_lat, nan=0.0, posinf=1.0, neginf=-1.0)
        out_denorm = (out_lat / scale) + shift
        image = self.pipe.vae.decode(out_denorm.to(self.pipe.vae.dtype), return_dict=False)[0]
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        return image


class AuraFlowAdapter(BaseAdapter):
    def _encode_src(self, image_pil: Image.Image, side: int):
        image = _pil_to_tensor01(image_pil, side=side).to(self.device)
        image = image * 2 - 1
        with torch.autocast("cuda"), torch.inference_mode():
            x_denorm = self.pipe.vae.encode(image.to(self.pipe.vae.dtype)).latent_dist.mode()
        scale = float(getattr(self.pipe.vae.config, "scaling_factor", 1.0))
        return (x_denorm * scale).to(self.model_dtype), scale

    @torch.no_grad()
    def edit(self, image_pil: Image.Image, src_prompt: str, tar_prompt: str, params: FlowEditParams) -> torch.Tensor:
        _seed_all(params.seed)
        x_src, scale = self._encode_src(image_pil, params.resize_side)
        scheduler = self.pipe.scheduler
        sigmas = np.linspace(1.0, 1.0 / params.diffusion_steps, params.diffusion_steps)
        timesteps, _ = retrieve_timesteps(scheduler, params.diffusion_steps, self.device, sigmas=sigmas)

        src = self.pipe.encode_prompt(
            prompt=src_prompt,
            negative_prompt=params.negative_prompt if params.negative_prompt else "",
            do_classifier_free_guidance=True,
            device=self.device,
        )
        tar = self.pipe.encode_prompt(
            prompt=tar_prompt,
            negative_prompt=params.negative_prompt if params.negative_prompt else "",
            do_classifier_free_guidance=True,
            device=self.device,
        )
        src_pos, _, src_neg, _ = src
        tar_pos, _, tar_neg, _ = tar
        src_pos = src_pos.to(self.model_dtype)
        src_neg = src_neg.to(self.model_dtype)
        tar_pos = tar_pos.to(self.model_dtype)
        tar_neg = tar_neg.to(self.model_dtype)

        def _velocity(latents, t, pos_emb, neg_emb, guidance):
            lat_in = torch.cat([latents, latents], dim=0)
            timestep = torch.tensor([t / 1000], device=self.device, dtype=latents.dtype).expand(lat_in.shape[0])
            enc = torch.cat([neg_emb, pos_emb], dim=0)
            pred = self.pipe.transformer(
                lat_in,
                encoder_hidden_states=enc,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=None,
            )[0]
            p_u, p_c = pred.chunk(2)
            return p_u + guidance * (p_c - p_u)

        out_lat = _run_flowedit_loop(
            x_src=x_src,
            timesteps=timesteps,
            scheduler=scheduler,
            params=params,
            velocity_src=lambda z, t: _velocity(z, t, src_pos, src_neg, params.src_guidance_scale),
            velocity_tar=lambda z, t: _velocity(z, t, tar_pos, tar_neg, params.tar_guidance_scale),
        )

        out_lat = torch.nan_to_num(out_lat, nan=0.0, posinf=1.0, neginf=-1.0)
        image = self.pipe.vae.decode((out_lat / scale).to(self.pipe.vae.dtype), return_dict=False)[0]
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        return image


def _load_sd3_single_file_from_repo(
    repo_id: str,
    cache_dir: str,
    hf_token: Optional[str],
):
    candidates = [
        "sd3_medium.safetensors",
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",
    ]

    errors = []
    # Prefer already-cached local snapshot files to avoid fragile network fetches.
    model_cache = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    for fn in candidates:
        if model_cache.exists():
            for p in model_cache.glob(f"*/{fn}"):
                if not p.exists():
                    continue
                try:
                    return StableDiffusion3Pipeline.from_single_file(str(p), torch_dtype=torch.float16)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{p.name}[local_snapshot]: {type(e).__name__}: {e}")

    for local_only in (True, False):
        for fn in candidates:
            try:
                ckpt = hf_hub_download(
                    repo_id=repo_id,
                    filename=fn,
                    cache_dir=cache_dir,
                    token=hf_token,
                    resume_download=True,
                    local_files_only=local_only,
                )
                pipe = StableDiffusion3Pipeline.from_single_file(ckpt, torch_dtype=torch.float16)
                return pipe
            except Exception as e:  # noqa: BLE001
                mode = "local_only" if local_only else "online"
                errors.append(f"{fn}[{mode}]: {type(e).__name__}: {e}")

    # Fallback for open SD3-medium: use diffusers-formatted repo when single-file
    # checkpoint is missing CLIP/T5 components.
    if repo_id == "stabilityai/stable-diffusion-3-medium":
        try:
            return StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16,
                token=hf_token,
                cache_dir=cache_dir,
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"fallback_diffusers: {type(e).__name__}: {e}")
    raise RuntimeError("Failed SD3 single-file load. " + " | ".join(errors))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
    except Exception:  # noqa: BLE001
        shutil.copy2(src, dst)


def _ensure_auraflow_fp16_compat(
    repo_id: str,
    cache_dir: str,
    hf_token: Optional[str],
) -> None:
    """Create compatibility aliases for deprecated fp16 shard naming.

    Some AuraFlow checkpoints use:
    - diffusion_pytorch_model-00001-of-00002.fp16.safetensors
    - diffusion_pytorch_model.safetensors.fp16.index.json

    while newer diffusers expects:
    - diffusion_pytorch_model.fp16-00001-of-00002.safetensors
    - diffusion_pytorch_model.fp16.safetensors.index.json
    """

    try:
        model_index = hf_hub_download(
            repo_id=repo_id,
            filename="model_index.json",
            cache_dir=cache_dir,
            token=hf_token,
            resume_download=True,
        )
    except Exception:  # noqa: BLE001
        return

    snap = Path(model_index).parent
    tr_dir = snap / "transformer"
    old_index = tr_dir / "diffusion_pytorch_model.safetensors.fp16.index.json"
    new_index = tr_dir / "diffusion_pytorch_model.fp16.safetensors.index.json"
    if not old_index.exists():
        return

    _link_or_copy(old_index, new_index)

    try:
        with old_index.open("r", encoding="utf-8") as f:
            old_index_json = json.load(f)
        shards = sorted(set(old_index_json.get("weight_map", {}).values()))
    except Exception:  # noqa: BLE001
        shards = []

    for shard in shards:
        if not shard.startswith("diffusion_pytorch_model-"):
            continue
        if not shard.endswith(".fp16.safetensors"):
            continue
        src = tr_dir / shard
        dst_name = shard.replace("diffusion_pytorch_model-", "diffusion_pytorch_model.fp16-", 1).replace(
            ".fp16.safetensors", ".safetensors"
        )
        dst = tr_dir / dst_name
        _link_or_copy(src, dst)


def _resolve_repo_snapshot(repo_id: str, cache_dir: str, hf_token: Optional[str]) -> Path:
    repo_path = Path(repo_id).expanduser()
    if repo_path.is_file():
        repo_path = repo_path.parent
    if repo_path.is_dir() and (repo_path / "model_index.json").is_file():
        return repo_path

    cache_root = Path(cache_dir)
    cached_repo = cache_root / f"models--{repo_id.replace('/', '--')}"
    ref_main = cached_repo / "refs" / "main"
    if ref_main.is_file():
        snapshot_id = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = cached_repo / "snapshots" / snapshot_id
        if (snapshot_dir / "model_index.json").is_file():
            return snapshot_dir

    snapshots_dir = cached_repo / "snapshots"
    if snapshots_dir.is_dir():
        for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True):
            if snapshot_dir.is_dir() and (snapshot_dir / "model_index.json").is_file():
                return snapshot_dir

    model_index = hf_hub_download(
        repo_id=repo_id,
        filename="model_index.json",
        cache_dir=cache_dir,
        token=hf_token,
        local_files_only=True,
    )
    return Path(model_index).parent


def _ensure_sd3_fp16_compat(snapshot_dir: Path) -> None:
    te3 = snapshot_dir / "text_encoder_3"
    old_te3_index = te3 / "model.safetensors.index.fp16.json"
    new_te3_index = te3 / "model.fp16.safetensors.index.json"
    if old_te3_index.exists():
        _link_or_copy(old_te3_index, new_te3_index)

    tr_dir = snapshot_dir / "transformer"
    tr_index = tr_dir / "diffusion_pytorch_model.safetensors.index.json"
    tr_fp16_index = tr_dir / "diffusion_pytorch_model.fp16.safetensors.index.json"
    if tr_index.exists():
        _link_or_copy(tr_index, tr_fp16_index)
        try:
            with tr_index.open("r", encoding="utf-8") as f:
                tr_index_json = json.load(f)
            tr_shards = sorted(set(tr_index_json.get("weight_map", {}).values()))
        except Exception:  # noqa: BLE001
            tr_shards = []
        for shard in tr_shards:
            if not shard.startswith("diffusion_pytorch_model-"):
                continue
            src = tr_dir / shard
            dst = tr_dir / shard.replace("diffusion_pytorch_model-", "diffusion_pytorch_model.fp16-", 1)
            _link_or_copy(src, dst)

    vae_dir = snapshot_dir / "vae"
    vae_src = vae_dir / "diffusion_pytorch_model.safetensors"
    vae_dst = vae_dir / "diffusion_pytorch_model.fp16.safetensors"
    if vae_src.exists():
        _link_or_copy(vae_src, vae_dst)


def _pick_local_index(model_dir: Path, stem: str, prefer_fp16: bool) -> Path:
    candidates = []
    if prefer_fp16:
        candidates.append(model_dir / f"{stem}.fp16.safetensors.index.json")
    candidates.append(model_dir / f"{stem}.safetensors.index.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No index json found for {model_dir} stem={stem}")


def _load_diffusers_model_from_shards(model_cls, model_dir: Path, torch_dtype: torch.dtype, device: torch.device):
    config, _unused = model_cls.load_config(model_dir, return_unused_kwargs=True)
    model = model_cls.from_config(config)
    model = model.to(device=device, dtype=torch_dtype)

    index_path = _pick_local_index(
        model_dir=model_dir,
        stem="diffusion_pytorch_model",
        prefer_fp16=(torch_dtype == torch.float16),
    )
    index_json = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = list(dict.fromkeys(index_json["weight_map"].values()))
    loaded_keys = set()

    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")
        target_device = str(device) if device.type == "cuda" else "cpu"
        state_dict = safetensors_load_file(str(shard_path), device=target_device)
        if torch_dtype != torch.float32:
            state_dict = {k: v.to(dtype=torch_dtype) for k, v in state_dict.items()}
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading {shard_path.name}: {unexpected[:8]}")
        loaded_keys.update(state_dict.keys())
        del state_dict
        gc.collect()

    expected_keys = set(index_json["weight_map"].keys())
    missing = sorted(expected_keys - loaded_keys)
    if missing:
        raise RuntimeError(f"Missing keys after shard load for {model_dir}: {missing[:8]}")
    return model


def _load_sd3_pipeline_manual(
    repo_id: str,
    cache_dir: str,
    hf_token: Optional[str],
    device: torch.device,
):
    from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    snapshot_dir = _resolve_repo_snapshot(repo_id=repo_id, cache_dir=cache_dir, hf_token=hf_token)
    _ensure_sd3_fp16_compat(snapshot_dir)

    load_dtype = torch.float16 if device.type == "cuda" else torch.float32

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot_dir / "scheduler")
    transformer = _load_diffusers_model_from_shards(
        model_cls=SD3Transformer2DModel,
        model_dir=snapshot_dir / "transformer",
        torch_dtype=load_dtype,
        device=device,
    )
    vae = AutoencoderKL.from_pretrained(
        snapshot_dir / "vae",
        torch_dtype=load_dtype,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        snapshot_dir / "text_encoder",
        torch_dtype=load_dtype,
        local_files_only=True,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        snapshot_dir / "text_encoder_2",
        torch_dtype=load_dtype,
        local_files_only=True,
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        snapshot_dir / "text_encoder_3",
        torch_dtype=load_dtype,
        local_files_only=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained(snapshot_dir / "tokenizer", local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(snapshot_dir / "tokenizer_2", local_files_only=True)
    tokenizer_3 = T5TokenizerFast.from_pretrained(snapshot_dir / "tokenizer_3", local_files_only=True)

    pipe = StableDiffusion3Pipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3,
    ).to(device)
    pipe.transformer.eval().requires_grad_(False)
    return pipe


def _load_sd3_pipeline_standard(
    repo_id: str,
    cache_dir: str,
    hf_token: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
):
    snapshot_dir = _resolve_repo_snapshot(repo_id=repo_id, cache_dir=cache_dir, hf_token=hf_token)
    _ensure_sd3_fp16_compat(snapshot_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        snapshot_dir,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).to(device)
    pipe.transformer.eval().requires_grad_(False)
    return pipe


def create_adapter(
    model_key: str,
    device: torch.device,
    dtype: torch.dtype,
    hf_token: Optional[str],
    cache_dir: str,
    override_model_id: Optional[str] = None,
) -> BaseAdapter:
    spec = get_model_spec(model_key)
    model_id = override_model_id if override_model_id else spec.model_id
    os.environ["HF_HOME"] = str(Path(cache_dir).parents[0])
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    if spec.loader == "flux1":
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return Flux1Adapter(pipe, device)

    if spec.loader == "flux2":
        # Requires access to official gated FLUX.2-dev.
        from diffusers import Flux2Pipeline

        # Load directly with a device_map to avoid peak-memory OOM from `.to(device)`.
        # Default to "balanced" so large FLUX.2 weights can be streamed safely.
        device_map = os.environ.get("FLOWEDIT_FLUX2_DEVICE_MAP", "balanced")
        pipe = Flux2Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
            cache_dir=cache_dir,
            device_map=device_map,
        )
        pipe.transformer.eval().requires_grad_(False)

        # Use the actual transformer device as adapter execution device.
        try:
            tr_device = next(pipe.transformer.parameters()).device
        except StopIteration:
            tr_device = device
        if tr_device.type != "cuda":
            raise RuntimeError(
                f"FLUX.2 transformer is on {tr_device}. "
                "Set FLOWEDIT_FLUX2_DEVICE_MAP=balanced and expose enough CUDA devices."
            )
        return Flux2KleinAdapter(pipe, torch.device(tr_device))

    if spec.loader == "flux2-klein":
        if (
            AutoencoderKLFlux2 is None
            or Flux2Transformer2DModel is None
            or Qwen2TokenizerFast is None
            or Qwen3ForCausalLM is None
            or Flux2KleinPipeline is None
        ):
            raise ImportError("Flux2-klein dependencies are unavailable in the current environment.")
        # Build manually to avoid diffusers from_pretrained annotation resolution
        # incompatibility on local custom pipeline classes (Python 3.9 + postponed annotations).
        is_distilled = False
        try:
            model_index = hf_hub_download(
                repo_id=model_id,
                filename="model_index.json",
                cache_dir=cache_dir,
                token=hf_token,
                resume_download=True,
            )
            with open(model_index, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            is_distilled = bool(cfg.get("is_distilled", False))
        except Exception:  # noqa: BLE001
            is_distilled = False

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            token=hf_token,
            cache_dir=cache_dir,
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            token=hf_token,
            cache_dir=cache_dir,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            token=hf_token,
            cache_dir=cache_dir,
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_id,
            subfolder="tokenizer",
            token=hf_token,
            cache_dir=cache_dir,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            token=hf_token,
            cache_dir=cache_dir,
        )

        pipe = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        ).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return Flux2KleinAdapter(pipe, device)

    if spec.loader == "sd3":
        prefer_manual = os.environ.get("EDITSPLAT_SD3_LOADER", "").strip().lower() == "manual"
        if prefer_manual:
            pipe = _load_sd3_pipeline_manual(
                repo_id=model_id,
                cache_dir=cache_dir,
                hf_token=hf_token,
                device=device,
            )
        else:
            try:
                pipe = _load_sd3_pipeline_standard(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                    device=device,
                    dtype=dtype,
                )
            except Exception:
                pipe = _load_sd3_pipeline_manual(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                    device=device,
                )
        return SD3Adapter(pipe, device)

    if spec.loader == "sd3-single-file":
        pipe = _load_sd3_single_file_from_repo(model_id, cache_dir=cache_dir, hf_token=hf_token).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return SD3Adapter(pipe, device)

    if spec.loader == "qwen-image-edit":
        if QwenImageEditPipeline is None:
            raise ImportError("QwenImageEditPipeline is unavailable in the current diffusers build.")
        pipe = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return QwenImageEditAdapter(pipe, device)

    if spec.loader == "z-image":
        if ZImagePipeline is None:
            raise ImportError("ZImagePipeline is unavailable in the current diffusers build.")
        pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return ZImageAdapter(pipe, device)

    if spec.loader == "auraflow":
        if AuraFlowPipeline is None:
            raise ImportError("AuraFlowPipeline is unavailable in the current diffusers build.")
        _ensure_auraflow_fp16_compat(repo_id=model_id, cache_dir=cache_dir, hf_token=hf_token)
        try:
            # Prefer fp16 variant to reduce download volume and speed up initialization.
            pipe = AuraFlowPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=cache_dir,
                variant="fp16",
            ).to(device)
        except Exception:
            pipe = AuraFlowPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=cache_dir,
            ).to(device)
        pipe.transformer.eval().requires_grad_(False)
        return AuraFlowAdapter(pipe, device)

    raise ValueError(f"Unknown loader: {spec.loader}")



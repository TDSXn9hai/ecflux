"""FLUX.2 Klein 相关的通用组件：模型加载、文本编码、Rectified Flow 加噪 / 采样、VAE 编解码。

约定（与 diffusers FlowMatchEulerDiscreteScheduler 一致）：
    x_t = (1 - t) * x0 + t * eps,   t ∈ [0, 1]，t=1 为纯噪声
    v   = eps - x0                   （网络预测的速度场）
报告中 "x_t = x0 + t·v" 为同一直线插值的简写。
"""

import logging
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, compute_empirical_mu

logger = logging.getLogger("ecflux")

# 复用官方管线中的纯函数（staticmethod），保证与推理管线的 token 布局 / 位置编码完全一致
patchify_latents = Flux2KleinPipeline._patchify_latents
unpatchify_latents = Flux2KleinPipeline._unpatchify_latents
pack_latents = Flux2KleinPipeline._pack_latents
unpack_latents_with_ids = Flux2KleinPipeline._unpack_latents_with_ids
prepare_latent_ids = Flux2KleinPipeline._prepare_latent_ids
prepare_text_ids = Flux2KleinPipeline._prepare_text_ids

PRUNE_SPEC_NAME = "prune_spec.json"


# ----------------------------------------------------------------------------- loading


def load_transformer(path: str, dtype=torch.bfloat16, subfolder: str | None = "transformer") -> Flux2Transformer2DModel:
    """加载原始或剪枝后的 DiT。

    - 原始模型：`path` 为 diffusers 管线目录 / HF repo（读取 transformer 子目录）。
    - 剪枝模型：`path` 目录内含 prune_spec.json + model.safetensors（由 save_pruned_transformer 生成）。
    """
    if os.path.isfile(os.path.join(path, PRUNE_SPEC_NAME)):
        from .pruning.prune_ops import load_pruned_transformer

        return load_pruned_transformer(path, dtype=dtype)
    if subfolder and os.path.isdir(path) and not os.path.isdir(os.path.join(path, subfolder)):
        subfolder = None
    return Flux2Transformer2DModel.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype)


def load_scheduler(pipeline_path: str | None) -> FlowMatchEulerDiscreteScheduler:
    if pipeline_path and os.path.isdir(os.path.join(pipeline_path, "scheduler")):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(pipeline_path, subfolder="scheduler")
    # 与 FLUX.2 Klein scheduler_config.json 一致的默认值
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=4096, time_shift_type="exponential",
    )


def load_vae(pipeline_path: str, dtype=torch.float32, device="cpu"):
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(pipeline_path, subfolder="vae", torch_dtype=dtype).to(device).eval()


class TextEncoder:
    """冻结的 Qwen3 文本编码器，输出与 Flux2KleinPipeline.encode_prompt 完全一致的 prompt_embeds。"""

    def __init__(self, pipeline_path: str, device="cuda", dtype=torch.bfloat16, max_sequence_length: int = 512,
                 hidden_states_layers=(9, 18, 27)):
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(pipeline_path, subfolder="tokenizer")
        self.model = Qwen3ForCausalLM.from_pretrained(pipeline_path, subfolder="text_encoder", torch_dtype=dtype)
        self.model.to(device).eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype
        self.max_sequence_length = max_sequence_length
        self.layers = tuple(hidden_states_layers)

    @torch.no_grad()
    def encode(self, prompts: list[str], return_mask: bool = False):
        embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            self.model, self.tokenizer, prompts, dtype=self.dtype, device=self.device,
            max_sequence_length=self.max_sequence_length, hidden_states_layers=self.layers,
        )
        if not return_mask:
            return embeds
        mask = self.attention_mask(prompts).to(self.device)
        return embeds, mask

    def attention_mask(self, prompts: list[str]) -> torch.Tensor:
        masks = []
        for p in prompts:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            m = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=self.max_sequence_length)["attention_mask"]
            masks.append(m)
        return torch.cat(masks, 0)

    def offload(self):
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def masked_mean_pool(embeds: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """(B, L, D) -> (B, D)，按有效 token 求平均，供复杂度判别器使用。"""
    if mask is None:
        return embeds.float().mean(1)
    m = mask.to(embeds.dtype).unsqueeze(-1)
    return (embeds.float() * m).sum(1) / m.sum(1).clamp_min(1.0)


# ----------------------------------------------------------------------------- rectified flow


def sample_timesteps(batch_size: int, mode: str = "uniform", device="cpu", generator=None,
                     logit_mean: float = 0.0, logit_std: float = 1.0, shift: float = 1.0) -> torch.Tensor:
    """训练时间步采样。报告 4.3.1 默认 t~U(0,1)；logit_normal / shift 作为与教师训练分布对齐的可选项。"""
    if mode == "uniform":
        t = torch.rand(batch_size, device=device, generator=generator)
    elif mode == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator) * logit_std + logit_mean)
    else:
        raise ValueError(f"unknown timestep mode {mode}")
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t.clamp(1e-4, 1.0 - 1e-4)


def add_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    """返回 (x_t, v_target)。x0 / noise 形状 (B, N, C)，t 形状 (B,)。"""
    tt = t.view(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
    x_t = (1.0 - tt) * x0 + tt * noise
    return x_t, noise - x0


def dit_forward(model, x_t, t, prompt_embeds, img_ids, txt_ids, joint_attention_kwargs=None) -> torch.Tensor:
    """DiT 单次前向，返回速度场预测 (B, N, C)。t ∈ [0,1]（与 pipeline 中 timestep/1000 一致）。"""
    return model(
        hidden_states=x_t, timestep=t, guidance=None, encoder_hidden_states=prompt_embeds,
        txt_ids=txt_ids, img_ids=img_ids, joint_attention_kwargs=joint_attention_kwargs, return_dict=False,
    )[0]


def make_ids(latents_bchw: torch.Tensor, prompt_embeds: torch.Tensor):
    return prepare_latent_ids(latents_bchw).to(latents_bchw.device), prepare_text_ids(prompt_embeds).to(latents_bchw.device)


@torch.no_grad()
def euler_sample(model, prompt_embeds, height: int, width: int, num_steps: int = 20, guidance_scale: float = 4.0,
                 negative_embeds=None, scheduler=None, generator=None, latent_channels: int | None = None,
                 vae_scale_factor: int = 8, return_trajectory: bool = False):
    """与 Flux2KleinPipeline 相同的 Euler 采样流程（不依赖文本编码器 / VAE），返回打包后的 latents 与 ids。"""
    device, dtype = prompt_embeds.device, next(model.parameters()).dtype
    bsz = prompt_embeds.shape[0]
    c = latent_channels or model.config.in_channels
    h = 2 * (int(height) // (vae_scale_factor * 2)) // 2
    w = 2 * (int(width) // (vae_scale_factor * 2)) // 2
    latents = torch.randn((bsz, c, h, w), generator=generator, device="cpu", dtype=torch.float32).to(device)
    img_ids = prepare_latent_ids(latents).to(device)
    txt_ids = prepare_text_ids(prompt_embeds).to(device)
    latents = pack_latents(latents)

    scheduler = scheduler or load_scheduler(None)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    scheduler.set_begin_index(0)
    use_cfg = guidance_scale > 1.0 and negative_embeds is not None
    if use_cfg:
        neg_txt_ids = prepare_text_ids(negative_embeds).to(device)

    traj = []
    for t in scheduler.timesteps:
        tt = (t.expand(bsz) / 1000).to(dtype)
        v = dit_forward(model, latents.to(dtype), tt, prompt_embeds.to(dtype), img_ids, txt_ids)
        if use_cfg:
            v_neg = dit_forward(model, latents.to(dtype), tt, negative_embeds.to(dtype), img_ids, neg_txt_ids)
            v = v_neg + guidance_scale * (v - v_neg)
        latents = scheduler.step(v.float(), t, latents.float(), return_dict=False)[0]
        if return_trajectory:
            traj.append(latents.clone())
    if return_trajectory:
        return latents, img_ids, traj
    return latents, img_ids


# ----------------------------------------------------------------------------- VAE


def _bn_stats(vae, ref: torch.Tensor):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(ref.device, ref.dtype)
    return mean, std


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    """images ∈ [-1,1], (B,3,H,W) -> 归一化后的 patchified latents (B, 4*C, H/16, W/16)，与 pipeline 一致。"""
    lat = vae.encode(images.to(vae.device, vae.dtype)).latent_dist.mode()
    lat = patchify_latents(lat)
    mean, std = _bn_stats(vae, lat)
    return (lat - mean) / std


@torch.no_grad()
def decode_latents(vae, packed: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
    """打包 latents -> 图像 ∈ [0,1], (B,3,H,W)。"""
    lat = unpack_latents_with_ids(packed, img_ids)
    mean, std = _bn_stats(vae, lat)
    lat = unpatchify_latents(lat * std + mean)
    img = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
    return (img.float() / 2 + 0.5).clamp(0, 1)


def to_pil(images: torch.Tensor):
    from PIL import Image

    arr = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    return [Image.fromarray(a) for a in arr]


# ----------------------------------------------------------------------------- tiny configs（冒烟测试用）


def tiny_transformer(num_layers=5, num_single_layers=20, heads=4, head_dim=16, joint_dim=32, in_channels=16,
                     mlp_ratio=3.0, seed=0) -> Flux2Transformer2DModel:
    """结构与 FLUX.2 Klein 4B 相同（5 Double + 20 Single），但维度极小，便于 CPU 冒烟测试。"""
    torch.manual_seed(seed)
    q = head_dim // 4
    return Flux2Transformer2DModel(
        patch_size=1, in_channels=in_channels, num_layers=num_layers, num_single_layers=num_single_layers,
        attention_head_dim=head_dim, num_attention_heads=heads, joint_attention_dim=joint_dim,
        timestep_guidance_channels=32, mlp_ratio=mlp_ratio, axes_dims_rope=(q, q, q, head_dim - 3 * q),
        rope_theta=2000, guidance_embeds=False,
    )


def prepare_fm_batch(batch: dict, device, dtype, generator=None, timestep_cfg: dict | None = None,
                     embeds_key: str = "prompt_embeds", text_encoder=None) -> dict:
    """把缓存 batch 变成一次 Rectified Flow 训练前向所需的全部输入。

    返回 dict: x_t, t, noise, x0, target(v), prompt_embeds, img_ids, txt_ids
    若缓存中没有 prompt_embeds，则用 text_encoder 在线编码 caption。
    """
    lat = batch["latents"].to(device=device, dtype=torch.float32)
    if embeds_key in batch:
        emb = batch[embeds_key].to(device=device, dtype=dtype)
    elif text_encoder is not None:
        emb = text_encoder.encode(batch["caption"]).to(device=device, dtype=dtype)
    else:
        raise KeyError(f"batch has no {embeds_key} and no text_encoder given")
    img_ids, txt_ids = make_ids(lat, emb)
    x0 = pack_latents(lat)
    if "noise" in batch and "t" in batch:  # 离线教师缓存：复用缓存时的噪声与时间步，保证师生输入一致
        noise = batch["noise"].to(device=device, dtype=torch.float32)
        t = batch["t"].to(device=device, dtype=torch.float32)
    else:
        noise = torch.randn(x0.shape, generator=generator, device="cpu", dtype=torch.float32).to(device)
        tc = timestep_cfg or {}
        t = sample_timesteps(x0.shape[0], mode=tc.get("mode", "uniform"), generator=generator,
                             logit_mean=tc.get("logit_mean", 0.0), logit_std=tc.get("logit_std", 1.0),
                             shift=tc.get("shift", 1.0)).to(device)
    x_t, v = add_noise(x0, t, noise)
    return {"x_t": x_t.to(dtype), "t": t.to(dtype), "noise": noise, "x0": x0, "target": v,
            "prompt_embeds": emb, "img_ids": img_ids, "txt_ids": txt_ids}


def fm_forward(model, fb: dict, x_t=None, prompt_embeds=None, txt_ids=None, **kw) -> torch.Tensor:
    return dit_forward(model, fb["x_t"] if x_t is None else x_t, fb["t"],
                       fb["prompt_embeds"] if prompt_embeds is None else prompt_embeds,
                       fb["img_ids"], fb["txt_ids"] if txt_ids is None else txt_ids, **kw)

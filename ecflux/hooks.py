"""特征抓取：隐藏层输出（4.3.2-2）、Double Block 跨模态注意力图（4.3.2-3）、逐头文本注意力质量（4.2.1 模态感知）。

通过替换注意力 processor + forward hook 实现，不修改 diffusers 源码。退出上下文后自动恢复。
块编号统一使用「原始模型」中的 0-based 下标；块剪枝后通过 model.ecflux_block_map 映射到当前 ModuleList 位置。
"""

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
    _get_qkv_projections,
)


def block_map(model) -> dict:
    """{'double': [原始下标...], 'single': [原始下标...]}，与当前 ModuleList 顺序一一对应。"""
    bm = getattr(model, "ecflux_block_map", None)
    if bm is None:
        bm = {"double": list(range(len(model.transformer_blocks))),
              "single": list(range(len(model.single_transformer_blocks)))}
    return bm


def resolve_block(model, kind: str, orig_idx: int):
    """返回 (module, 当前位置)；若该块已被剪除则返回 (None, None)。"""
    bm = block_map(model)[kind]
    if orig_idx not in bm:
        return None, None
    pos = bm.index(orig_idx)
    blocks = model.transformer_blocks if kind == "double" else model.single_transformer_blocks
    return blocks[pos], pos


def _txt_attention_probs(q, k, txt_len: int, query_slice: slice, stride: int):
    """q,k: (B,S,H,D)。返回图像 query 对文本 key 的 softmax 注意力 (B,H,Nq,T)；softmax 在全部 key 上归一化。"""
    qi = q[:, query_slice][:, ::stride].transpose(1, 2).float()  # B,H,Nq,D
    kk = k.transpose(1, 2).float()  # B,H,S,D
    logits = qi @ kk.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    return logits.softmax(-1)[..., :txt_len]


class _RecordingDoubleProcessor(Flux2AttnProcessor):
    def __init__(self, recorder, name, want_map: bool, want_mass: bool):
        super().__init__()
        self.rec, self.name, self.want_map, self.want_mass = recorder, name, want_map, want_mass

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
        ev = ev.unflatten(-1, (attn.heads, -1))
        query, key, value = torch.cat([eq, query], 1), torch.cat([ek, key], 1), torch.cat([ev, value], 1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        t = encoder_hidden_states.shape[1]
        self.rec._record_attn(self.name, query, key, t, self.want_map, self.want_mass)

        out = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend,
                                    parallel_config=self._parallel_config)
        out = out.flatten(2, 3).to(query.dtype)
        enc, out = out.split_with_sizes([t, out.shape[1] - t], dim=1)
        return attn.to_out[1](attn.to_out[0](out)), attn.to_add_out(enc)


class _RecordingSingleProcessor(Flux2ParallelSelfAttnProcessor):
    def __init__(self, recorder, name):
        super().__init__()
        self.rec, self.name = recorder, name

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        hs = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = torch.split(hs, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in qkv.chunk(3, dim=-1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        self.rec._record_attn(self.name, q, k, self.rec.txt_len, False, True)
        out = dispatch_attention_fn(q, k, v, attn_mask=attention_mask, backend=self._attention_backend,
                                    parallel_config=self._parallel_config)
        out = out.flatten(2, 3).to(q.dtype)
        return attn.to_out(torch.cat([out, attn.mlp_act_fn(mlp)], dim=-1))


class FeatureRecorder:
    """上下文管理器。

    hidden_layers : ["double:3", "single:11", ...]  记录该块输出的图像 token 隐藏状态 (B, N_img, D)
    attn_layers   : [0, 1, ...]  记录这些 Double Block 的图像→文本注意力 (B, H, Nq, T)
    head_mass     : True 时记录所有块逐头的「图像 query 落在文本 key 上的注意力质量」(H,)，用于头类型判定
    query_stride  : 注意力图的图像 query 下采样步长，降低显存
    """

    def __init__(self, model, hidden_layers=(), attn_layers=(), head_mass: bool = False, query_stride: int = 1,
                 detach: bool = False):
        self.model = model
        self.hidden_layers = list(hidden_layers)
        self.attn_layers = list(attn_layers)
        self.head_mass_on = head_mass
        self.query_stride = max(1, int(query_stride))
        self.detach = detach
        self.hidden, self.attn, self.head_mass = {}, {}, {}
        self.txt_len = None
        self.txt_mask = None  # (B, T)；设置后 head_mass 只统计非 padding 文本 token
        self._handles, self._saved_proc = [], []

    # -- recording callbacks
    def _record_attn(self, name, q, k, txt_len, want_map, want_mass):
        img = slice(txt_len, None)
        if want_map:
            p = _txt_attention_probs(q, k, txt_len, img, self.query_stride)
            self.attn[name] = p.detach() if self.detach else p
        if want_mass and self.head_mass_on:
            with torch.no_grad():
                p = _txt_attention_probs(q, k, txt_len, img, max(self.query_stride, 4))
                if self.txt_mask is not None:
                    p = p * self.txt_mask.to(p.device, p.dtype)[:, None, None, :]
                m = p.sum(-1).mean(dim=(0, 2))  # H
            key = name if isinstance(name, tuple) else ("double", name)
            self.head_mass[key] = self.head_mass.get(key, 0) + m.cpu()

    def _hidden_hook(self, name, kind):
        def hook(_mod, _inp, out):
            if kind == "double":
                h = out[1]
            else:
                h = out[1] if isinstance(out, tuple) else out[:, self.txt_len:]
            self.hidden[name] = h.detach() if self.detach else h

        return hook

    def __enter__(self):
        m = self.model

        def pre(_mod, args, kwargs):
            ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
            self.txt_len = ehs.shape[1]

        self._handles.append(m.register_forward_pre_hook(pre, with_kwargs=True))

        for spec in self.hidden_layers:
            kind, idx = spec.split(":")
            blk, _ = resolve_block(m, kind, int(idx))
            if blk is None:
                raise ValueError(f"hidden layer {spec} has been pruned away")
            self._handles.append(blk.register_forward_hook(self._hidden_hook(spec, kind)))

        bm = block_map(m)
        for pos, orig in enumerate(bm["double"]):
            want_map = orig in self.attn_layers
            if want_map or self.head_mass_on:
                attn = m.transformer_blocks[pos].attn
                self._saved_proc.append((attn, attn.processor))
                attn.set_processor(_RecordingDoubleProcessor(self, orig, want_map, self.head_mass_on))
        if self.head_mass_on:
            for pos, orig in enumerate(bm["single"]):
                attn = m.single_transformer_blocks[pos].attn
                self._saved_proc.append((attn, attn.processor))
                attn.set_processor(_RecordingSingleProcessor(self, ("single", orig)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        for attn, proc in self._saved_proc:
            attn.set_processor(proc)
        self._handles, self._saved_proc = [], []

    def clear(self):
        self.hidden, self.attn = {}, {}

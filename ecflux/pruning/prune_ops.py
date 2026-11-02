"""结构化剪枝的物理操作：真正删除注意力头 / FFN 神经元 / Single Block 的参数，得到更小的稠密模型。

FLUX.2 Klein 的参数布局（diffusers 0.37）：
- Double Block: attn.to_q/k/v (图像流) 与 attn.add_q/k/v_proj (文本流) 按 head 切行；attn.to_out[0] / to_add_out 按 head 切列；
  ff / ff_context 为 SwiGLU：linear_in 输出 [gate | up] 两半，linear_out 按神经元切列。
- Single Block: attn.to_qkv_mlp_proj 输出 [q | k | v | mlp_gate | mlp_up]；attn.to_out 输入 [attn | mlp]。
QK-RMSNorm 作用在 head_dim 上，与头数无关，无需修改。
"""

import copy
import json
import os

import torch
import torch.nn as nn

from ..flux_utils import PRUNE_SPEC_NAME
from ..hooks import block_map


def _new_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin


def keep_rows(lin: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    idx = idx.to(lin.weight.device)
    return _new_linear(lin.weight.data[idx], None if lin.bias is None else lin.bias.data[idx])


def keep_cols(lin: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    idx = idx.to(lin.weight.device)
    return _new_linear(lin.weight.data[:, idx], None if lin.bias is None else lin.bias.data)


def head_rows(heads: torch.Tensor, head_dim: int) -> torch.Tensor:
    heads = torch.as_tensor(heads, dtype=torch.long).sort().values
    return (heads[:, None] * head_dim + torch.arange(head_dim)[None]).flatten()


# ----------------------------------------------------------------------------- double block


def prune_double_heads(block, keep_heads):
    attn = block.attn
    rows = head_rows(keep_heads, attn.head_dim)
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        setattr(attn, name, keep_rows(getattr(attn, name), rows))
    attn.to_out[0] = keep_cols(attn.to_out[0], rows)
    attn.to_add_out = keep_cols(attn.to_add_out, rows)
    attn.heads = len(keep_heads)
    attn.inner_dim = attn.heads * attn.head_dim


def prune_swiglu_ffn(ff, keep):
    """Flux2FeedForward：linear_in 行 [gate(0..M) | up(M..2M)]，linear_out 列 M。"""
    keep = torch.as_tensor(keep, dtype=torch.long).sort().values
    m = ff.linear_out.in_features
    ff.linear_in = keep_rows(ff.linear_in, torch.cat([keep, keep + m]))
    ff.linear_out = keep_cols(ff.linear_out, keep)


# ----------------------------------------------------------------------------- single block


def _single_layout(attn):
    i, m = attn.inner_dim, attn.mlp_hidden_dim
    return i, m


def prune_single_block(block, keep_heads=None, keep_mlp=None):
    """同时（或分别）裁剪 Single Block 的头与 MLP 神经元，保证 fused 投影布局一致。"""
    attn = block.attn
    inner, m = _single_layout(attn)
    hd = attn.head_dim
    if keep_heads is None:
        keep_heads = torch.arange(attn.heads)
    if keep_mlp is None:
        keep_mlp = torch.arange(m)
    keep_mlp = torch.as_tensor(keep_mlp, dtype=torch.long).sort().values
    hr = head_rows(keep_heads, hd)
    in_rows = torch.cat([hr, hr + inner, hr + 2 * inner, 3 * inner + keep_mlp, 3 * inner + m + keep_mlp])
    out_cols = torch.cat([hr, inner + keep_mlp])
    attn.to_qkv_mlp_proj = keep_rows(attn.to_qkv_mlp_proj, in_rows)
    attn.to_out = keep_cols(attn.to_out, out_cols)
    attn.heads = len(keep_heads)
    attn.inner_dim = attn.heads * hd
    attn.mlp_hidden_dim = len(keep_mlp)


# ----------------------------------------------------------------------------- block removal


def remove_single_blocks(model, orig_indices):
    """删除指定（原始下标）的 Single Block。Double Block 承担双流交互，按报告设定不做块级删除。"""
    bm = block_map(model)
    drop = set(int(i) for i in orig_indices)
    keep_pos = [p for p, o in enumerate(bm["single"]) if o not in drop]
    model.single_transformer_blocks = nn.ModuleList([model.single_transformer_blocks[p] for p in keep_pos])
    model.ecflux_block_map = {"double": list(bm["double"]), "single": [bm["single"][p] for p in keep_pos]}


# ----------------------------------------------------------------------------- spec / save / load


def structure_spec(model) -> dict:
    """描述当前（可能已剪枝）模型的结构，足以从原始 config 重建同形状网络。"""
    bm = block_map(model)
    spec = {"base_config": {k: v for k, v in dict(model.config).items() if not k.startswith("_")},
            "double": [], "single": []}
    for pos, orig in enumerate(bm["double"]):
        b = model.transformer_blocks[pos]
        spec["double"].append({"orig": orig, "heads": b.attn.heads, "ff": b.ff.linear_out.in_features,
                               "ff_context": b.ff_context.linear_out.in_features})
    for pos, orig in enumerate(bm["single"]):
        b = model.single_transformer_blocks[pos]
        spec["single"].append({"orig": orig, "heads": b.attn.heads, "mlp": b.attn.mlp_hidden_dim})
    units = getattr(model, "ecflux_units", None)
    if units:  # 保留单元在原始模型中的编号（逐头对齐蒸馏 / 分析时使用）
        for kind in ("double", "single"):
            for s in spec[kind]:
                u = units.get((kind, s["orig"]))
                if u:
                    s["units"] = {k: v for k, v in u.items() if k != "head_type"}
    return spec


def apply_structure_spec(model, spec: dict):
    """把一个「原始结构」模型按 spec 收缩形状（保留前 k 个单元，随后由 state_dict 覆盖权重）。"""
    keep_single = {s["orig"] for s in spec["single"]}
    remove_single_blocks(model, [o for o in block_map(model)["single"] if o not in keep_single])
    for pos, s in enumerate(spec["double"]):
        b = model.transformer_blocks[pos]
        prune_double_heads(b, torch.arange(s["heads"]))
        prune_swiglu_ffn(b.ff, torch.arange(s["ff"]))
        prune_swiglu_ffn(b.ff_context, torch.arange(s["ff_context"]))
    for pos, s in enumerate(spec["single"]):
        prune_single_block(model.single_transformer_blocks[pos], torch.arange(s["heads"]), torch.arange(s["mlp"]))
    if all("units" in s for k in ("double", "single") for s in spec[k]):
        model.ecflux_units = {(k, s["orig"]): dict(s["units"]) for k in ("double", "single") for s in spec[k]}
    return model


def save_pruned_transformer(model, out_dir: str, extra: dict | None = None):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    spec = structure_spec(model)
    if extra:
        spec["extra"] = extra
    with open(os.path.join(out_dir, PRUNE_SPEC_NAME), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))


def load_pruned_transformer(path: str, dtype=torch.bfloat16):
    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import load_file

    with open(os.path.join(path, PRUNE_SPEC_NAME), encoding="utf-8") as f:
        spec = json.load(f)
    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(spec["base_config"])
    apply_structure_spec(model, spec)
    sd = load_file(os.path.join(path, "model.safetensors"))
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(dtype)


def clone_model(model):
    """深拷贝（含 ecflux_block_map 属性），用于 APRA 回滚快照。"""
    return copy.deepcopy(model)

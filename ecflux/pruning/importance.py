"""4.2.1 多粒度重要性评估。

- 注意力头：梯度敏感度 S_head^i = |∂L/∂W_Q^i ⊙ W_Q^i|_1 + K + V（Double Block 同时累加文本流 add_q/k/v_proj）
- 头类型：逐头统计「图像 query 投向文本 key 的注意力质量」，相对均匀分布的倍数 > 阈值即判为跨模态头
- FFN 神经元：SwiGLU 激活值 L1 均值 S_neuron^j = 1/N Σ|a_j|
- Block：相邻 Single Block 输出的 token 级余弦相似度均值

消融开关：head_metric ∈ {grad, magnitude, random}，neuron_metric ∈ {activation, magnitude, random}
"""

import logging

import torch
import torch.nn.functional as F

from ..flux_utils import fm_forward
from ..hooks import FeatureRecorder, block_map

logger = logging.getLogger("ecflux")


def _head_reduce(x: torch.Tensor, heads: int) -> torch.Tensor:
    """(heads*head_dim, in) 的逐元素分数 -> (heads,)"""
    return x.float().abs().view(heads, -1).sum(-1)


def _attn_param_groups(model):
    """返回 [(kind, orig, [(param, row_offset_fn)...])]，用于按头累加分数。"""
    out = []
    bm = block_map(model)
    for pos, orig in enumerate(bm["double"]):
        a = model.transformer_blocks[pos].attn
        out.append(("double", orig, a, [a.to_q.weight, a.to_k.weight, a.to_v.weight,
                                         a.add_q_proj.weight, a.add_k_proj.weight, a.add_v_proj.weight]))
    for pos, orig in enumerate(bm["single"]):
        a = model.single_transformer_blocks[pos].attn
        out.append(("single", orig, a, [a.to_qkv_mlp_proj.weight]))
    return out


def _head_scores_from(kind, attn, tensors) -> torch.Tensor:
    """tensors 与 _attn_param_groups 中参数一一对应（可以是 grad*W、W 等）。"""
    h = attn.heads
    if kind == "double":
        return sum(_head_reduce(t, h) for t in tensors)
    inner = attn.inner_dim
    t = tensors[0]
    return sum(_head_reduce(t[i * inner:(i + 1) * inner], h) for i in range(3))


def head_grad_sensitivity(model, batches, loss_fn=None) -> dict:
    """在校准 batch 上做前向+反向，仅为 QKV 相关参数保留梯度，累加 |g ⊙ W|。"""
    groups = _attn_param_groups(model)
    target_params = {id(p) for _, _, _, ps in groups for p in ps}
    saved = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in target_params)
        p.grad = None
    acc = {(k, o): 0 for k, o, _, _ in groups}
    model.train(False)
    for fb in batches:
        pred = fm_forward(model, fb)
        loss = loss_fn(pred, fb) if loss_fn else F.mse_loss(pred.float(), fb["target"].float())
        loss.backward()
        with torch.no_grad():
            for kind, orig, attn, ps in groups:
                acc[(kind, orig)] = acc[(kind, orig)] + _head_scores_from(kind, attn, [p.grad * p for p in ps]).cpu()
                for p in ps:
                    p.grad = None
    for n, p in model.named_parameters():
        p.requires_grad_(saved[n])
    return acc


@torch.no_grad()
def head_magnitude(model) -> dict:
    return {(k, o): _head_scores_from(k, a, ps).cpu() for k, o, a, ps in _attn_param_groups(model)}


class _ActStats:
    """SwiGLU 输出 a_j 的 L1 均值累加器（forward hook）。"""

    def __init__(self, name: str = "ff", ctx: dict | None = None):
        self.sum, self.n = 0, 0
        self.name, self.ctx = name, ctx if ctx is not None else {}

    def _weights(self, a):
        """按 token 加权：ctx["mask"] (B,T) 非空时排除 padding 文本 token（ff_context 全是文本，single 前 T 个是文本）。"""
        m = self.ctx.get("mask")
        if m is None or self.name == "ff":
            return None
        m = m.to(a.device, a.dtype)
        if self.name == "mlp":
            m = torch.cat([m, m.new_ones(m.shape[0], a.shape[1] - m.shape[1])], 1)
        return m

    def __call__(self, _mod, _inp, out):
        a = out.detach().float().abs()
        w = self._weights(a)
        if w is None:
            self.sum = self.sum + a.reshape(-1, a.shape[-1]).sum(0).cpu()
            self.n += a.reshape(-1, a.shape[-1]).shape[0]
        else:
            self.sum = self.sum + (a * w[..., None]).sum((0, 1)).cpu()
            self.n += float(w.sum())

    @property
    def value(self):
        return self.sum / max(1, self.n)


def _ffn_modules(model):
    bm = block_map(model)
    mods = []
    for pos, orig in enumerate(bm["double"]):
        b = model.transformer_blocks[pos]
        mods.append((("double", orig, "ff"), b.ff.act_fn, b.ff.linear_out))
        mods.append((("double", orig, "ff_context"), b.ff_context.act_fn, b.ff_context.linear_out))
    for pos, orig in enumerate(bm["single"]):
        a = model.single_transformer_blocks[pos].attn
        mods.append((("single", orig, "mlp"), a.mlp_act_fn, None))
    return mods


@torch.no_grad()
def forward_statistics(model, batches, want_neurons=True, want_mass=True, want_block_sim=True,
                       mask_padding=True, neuron_mask_padding=False) -> dict:
    """一次无梯度遍历同时统计：神经元激活 L1、逐头文本注意力质量、相邻 Single Block 余弦相似度。"""
    stats, handles = {}, []
    act_ctx = {"mask": None}
    if want_neurons:
        for key, act, _ in _ffn_modules(model):
            stats[key] = _ActStats(key[2], act_ctx)
            handles.append(act.register_forward_hook(stats[key]))

    bm = block_map(model)
    outs = {}
    sim_sum, sim_n = {}, 0
    if want_block_sim:
        for pos, orig in enumerate(bm["single"]):
            def hook(_m, _i, o, orig=orig):
                outs[orig] = o.detach()
            handles.append(model.single_transformer_blocks[pos].register_forward_hook(hook))

    rec = FeatureRecorder(model, head_mass=want_mass, query_stride=4, detach=True)
    mass_norm = {}
    try:
        with rec:
            for fb in batches:
                outs.clear()
                rec.txt_mask = fb.get("txt_mask") if mask_padding else None
                act_ctx["mask"] = fb.get("txt_mask") if neuron_mask_padding else None
                fm_forward(model, fb)
                if want_mass:
                    t, n = fb["prompt_embeds"].shape[1], fb["x_t"].shape[1]
                    if rec.txt_mask is not None:  # 均匀分配基准只按有效文本 token 计
                        t = float(rec.txt_mask.float().sum(-1).mean())
                    mass_norm["uniform"] = t / (t + n)
                if want_block_sim:
                    order = bm["single"]
                    for a, b in zip(order[:-1], order[1:]):
                        s = F.cosine_similarity(outs[a].float(), outs[b].float(), dim=-1).mean().item()
                        sim_sum[(a, b)] = sim_sum.get((a, b), 0.0) + s
                    sim_n += 1
    finally:
        for h in handles:
            h.remove()

    res = {"neurons": {k: v.value for k, v in stats.items()}}
    if want_mass:
        nb = max(1, len(batches))
        uni = mass_norm.get("uniform", 1.0)
        # 相对均匀分配的倍数：>1 表示该头偏向文本 token（跨模态绑定头）
        res["head_text_ratio"] = {k: (v / nb) / uni for k, v in rec.head_mass.items()}
    if want_block_sim:
        res["block_sim"] = {k: v / max(1, sim_n) for k, v in sim_sum.items()}
    return res


def neuron_magnitude(model) -> dict:
    out = {}
    bm = block_map(model)
    with torch.no_grad():
        for pos, orig in enumerate(bm["double"]):
            b = model.transformer_blocks[pos]
            for name in ("ff", "ff_context"):
                ff = getattr(b, name)
                m = ff.linear_out.in_features
                w_in = ff.linear_in.weight.float()
                out[("double", orig, name)] = (w_in[:m].abs().sum(1) * w_in[m:].abs().sum(1)).sqrt().cpu() \
                    * ff.linear_out.weight.float().abs().sum(0).cpu()
        for pos, orig in enumerate(bm["single"]):
            a = model.single_transformer_blocks[pos].attn
            i, m = a.inner_dim, a.mlp_hidden_dim
            w_in = a.to_qkv_mlp_proj.weight.float()
            g, u = w_in[3 * i:3 * i + m], w_in[3 * i + m:3 * i + 2 * m]
            out[("single", orig, "mlp")] = ((g.abs().sum(1) * u.abs().sum(1)).sqrt()
                                            * a.to_out.weight.float()[:, i:].abs().sum(0)).cpu()
    return out


@torch.no_grad()
def head_ablation_scores(model, batches, kinds=("double",)) -> dict:
    """逐头删除后的速度场误差（相对当前模型，×100）作为头重要性，值越大越重要。
    阶段1发现梯度敏感度会把 double:0 的关键头（删掉单个头即造成生成崩溃）排在末尾，Double Block 头数少，可直接逐头测。"""
    import copy

    from .prune_ops import prune_double_heads

    ref = [fm_forward(model, fb).float() for fb in batches]
    bm = block_map(model)
    out = {}
    if "double" in kinds:
        for pos, orig in enumerate(bm["double"]):
            blk = model.transformer_blocks[pos]
            h = blk.attn.heads
            scores = torch.zeros(h)
            for i in range(h if h > 1 else 0):
                tmp = copy.deepcopy(blk)
                prune_double_heads(tmp, torch.tensor([j for j in range(h) if j != i]))
                model.transformer_blocks[pos] = tmp
                scores[i] = sum(100 * F.mse_loss(fm_forward(model, fb).float(), r).item()
                                for fb, r in zip(batches, ref)) / len(batches)
                model.transformer_blocks[pos] = blk
                del tmp
            out[("double", orig)] = scores
    return out


def collect_importance(model, batches, cfg: dict, seed: int = 0) -> dict:
    """汇总三粒度重要性。返回
    {"heads": {(kind,orig): (H,)}, "head_text_ratio": {...}, "neurons": {(kind,orig,name): (M,)}, "block_sim": {...}}
    """
    head_metric = cfg.get("head_metric", "grad")
    neuron_metric = cfg.get("neuron_metric", "activation")
    fs = forward_statistics(model, batches, want_neurons=neuron_metric == "activation",
                            want_mass=cfg.get("modality_aware", True), want_block_sim=True,
                            mask_padding=cfg.get("mask_text_padding", True),
                            neuron_mask_padding=cfg.get("neuron_mask_padding", False))
    g = torch.Generator().manual_seed(seed)
    if head_metric == "grad":
        heads = head_grad_sensitivity(model, batches)
    elif head_metric == "magnitude":
        heads = head_magnitude(model)
    elif head_metric == "random":
        heads = {k: torch.rand(v.shape, generator=g) for k, v in head_magnitude(model).items()}
    else:
        raise ValueError(head_metric)
    if head_metric == "grad" and cfg.get("double_head_metric", "grad") == "ablation":
        heads.update(head_ablation_scores(model, batches[: int(cfg.get("head_ablation_batches", 8))]))
    if neuron_metric == "activation":
        neurons = fs["neurons"]
    elif neuron_metric == "magnitude":
        neurons = neuron_magnitude(model)
    elif neuron_metric == "random":
        neurons = {k: torch.rand(v.shape, generator=g) for k, v in neuron_magnitude(model).items()}
    else:
        raise ValueError(neuron_metric)
    return {"heads": heads, "head_text_ratio": fs.get("head_text_ratio", {}), "neurons": neurons,
            "block_sim": fs["block_sim"]}

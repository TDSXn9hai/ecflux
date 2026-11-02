"""三粒度剪枝的规划与执行（4.2.1 + 表1 层级组）。

剪枝率 γ_g 始终相对「原始模型」的单元数定义：第 t 轮把每层剪到 floor(γ_g · 原始数量)，已剪掉的单元不会恢复。
模态感知约束：跨模态头最多剪掉 α_cross，自注意力头最多剪掉 α_self；FFN 神经元剪枝率上界 β_ffn。
"""

import logging
import math

import torch

from ..hooks import block_map
from . import prune_ops

logger = logging.getLogger("ecflux")

DEFAULT_GROUPS = [  # 报告表1
    {"name": "G1", "double": [0, 1], "single": [], "init": 0.05},
    {"name": "G2", "double": [2, 3], "single": [], "init": 0.10},
    {"name": "G3", "double": [4], "single": [], "init": 0.15},
    {"name": "G4", "double": [], "single": list(range(0, 8)), "init": 0.15},
    {"name": "G5", "double": [], "single": list(range(8, 16)), "init": 0.25},
    {"name": "G6", "double": [], "single": list(range(16, 20)), "init": 0.35},
]


def group_of(groups, kind: str, orig: int) -> str | None:
    for g in groups:
        if orig in g.get(kind, []):
            return g["name"]
    return None


def init_unit_meta(model):
    """记录每个块当前保留单元对应的原始编号；首次调用时建立。"""
    if getattr(model, "ecflux_units", None) is not None:
        return model.ecflux_units
    units = {}
    bm = block_map(model)
    for pos, orig in enumerate(bm["double"]):
        b = model.transformer_blocks[pos]
        units[("double", orig)] = {"heads": list(range(b.attn.heads)), "ff": list(range(b.ff.linear_out.in_features)),
                                   "ff_context": list(range(b.ff_context.linear_out.in_features)),
                                   "orig_heads": b.attn.heads, "orig_ff": b.ff.linear_out.in_features}
    for pos, orig in enumerate(bm["single"]):
        a = model.single_transformer_blocks[pos].attn
        units[("single", orig)] = {"heads": list(range(a.heads)), "mlp": list(range(a.mlp_hidden_dim)),
                                   "orig_heads": a.heads, "orig_mlp": a.mlp_hidden_dim}
    model.ecflux_units = units
    return units


def assign_head_types(model, head_text_ratio: dict, threshold: float):
    """首次剪枝前根据原始模型的注意力统计确定每个原始头的类型（cross / self），之后保持不变。"""
    units = init_unit_meta(model)
    for key, u in units.items():
        if "head_type" in u:
            continue
        r = head_text_ratio.get(key)
        if r is None:
            u["head_type"] = {h: "self" for h in u["heads"]}
        else:
            u["head_type"] = {h: ("cross" if float(r[i]) > threshold else "self") for i, h in enumerate(u["heads"])}


def _select_heads(u, scores, n_target_pruned, cfg) -> list[int]:
    """返回需要保留的「当前位置」下标。"""
    cur = u["heads"]
    n_orig = u["orig_heads"]
    already = n_orig - len(cur)
    extra = n_target_pruned - already
    if extra <= 0:
        return list(range(len(cur)))
    types = u.get("head_type", {h: "self" for h in cur})
    if cfg.get("modality_aware", True):
        caps = {"cross": cfg.get("alpha_cross", 0.15), "self": cfg.get("alpha_self", 0.40)}
    else:
        caps = {"cross": cfg.get("alpha_self", 0.40), "self": cfg.get("alpha_self", 0.40)}
    orig_cnt = {t: sum(1 for v in types.values() if v == t) for t in ("cross", "self")}
    cur_cnt = {t: sum(1 for h in cur if types[h] == t) for t in ("cross", "self")}
    budget = {t: math.floor(caps[t] * orig_cnt[t]) - (orig_cnt[t] - cur_cnt[t]) for t in caps}
    min_heads = int(cfg.get("min_heads", 1))
    drop = set()
    for pos in torch.argsort(scores).tolist():
        if len(drop) >= extra or len(cur) - len(drop) <= min_heads:
            break
        t = types[cur[pos]]
        if budget[t] > 0:
            drop.add(pos)
            budget[t] -= 1
    return [p for p in range(len(cur)) if p not in drop]


def _select_neurons(cur_list, n_orig, scores, rate, beta, align: int = 8) -> list[int]:
    # 剪枝数取 align 的整数倍：保留维度非 8 的倍数时，Single Block 融合投影的行跨度不对齐，
    # SDPA 反向（梯度检查点重算）会报 misaligned address，且张量核利用率下降
    n_target_pruned = math.floor(min(rate, beta) * n_orig / align) * align
    extra = n_target_pruned - (n_orig - len(cur_list))
    if extra <= 0:
        return list(range(len(cur_list)))
    drop = set(torch.argsort(scores)[:extra].tolist())
    return [p for p in range(len(cur_list)) if p not in drop]


def prune_to_rates(model, importance: dict, rates: dict, groups: list, cfg: dict) -> dict:
    """按各组剪枝率执行头 + 神经元剪枝。cfg 中 prune_heads / prune_neurons 为粒度消融开关。"""
    units = init_unit_meta(model)
    if cfg.get("modality_aware", True):
        assign_head_types(model, importance.get("head_text_ratio", {}), cfg.get("cross_ratio_threshold", 1.0))
    beta = cfg.get("beta_ffn", 0.35)
    align = int(cfg.get("neuron_align", 8))
    alpha_max = max(cfg.get("alpha_self", 0.40), cfg.get("alpha_cross", 0.15))
    bm = block_map(model)
    report = {}
    for kind in ("double", "single"):
        blocks = model.transformer_blocks if kind == "double" else model.single_transformer_blocks
        for pos, orig in enumerate(bm[kind]):
            g = group_of(groups, kind, orig)
            if g is None:
                continue
            rate = rates[g]
            u = units[(kind, orig)]
            blk = blocks[pos]
            keep_h = list(range(len(u["heads"])))
            if cfg.get("prune_heads", True):
                n_t = math.floor(min(rate, alpha_max) * u["orig_heads"])
                keep_h = _select_heads(u, importance["heads"][(kind, orig)], n_t, cfg)
            if kind == "double":
                keep_ff = list(range(len(u["ff"])))
                keep_fc = list(range(len(u["ff_context"])))
                if cfg.get("prune_neurons", True):
                    keep_ff = _select_neurons(u["ff"], u["orig_ff"], importance["neurons"][(kind, orig, "ff")], rate, beta, align)
                    if cfg.get("prune_ff_context", True):
                        keep_fc = _select_neurons(u["ff_context"], u["orig_ff"],
                                                  importance["neurons"][(kind, orig, "ff_context")], rate, beta, align)
                if len(keep_h) < len(u["heads"]):
                    prune_ops.prune_double_heads(blk, torch.tensor(keep_h))
                if len(keep_ff) < len(u["ff"]):
                    prune_ops.prune_swiglu_ffn(blk.ff, torch.tensor(keep_ff))
                if len(keep_fc) < len(u["ff_context"]):
                    prune_ops.prune_swiglu_ffn(blk.ff_context, torch.tensor(keep_fc))
                u["ff"] = [u["ff"][i] for i in keep_ff]
                u["ff_context"] = [u["ff_context"][i] for i in keep_fc]
            else:
                keep_m = list(range(len(u["mlp"])))
                if cfg.get("prune_neurons", True):
                    keep_m = _select_neurons(u["mlp"], u["orig_mlp"], importance["neurons"][(kind, orig, "mlp")], rate, beta, align)
                if len(keep_h) < len(u["heads"]) or len(keep_m) < len(u["mlp"]):
                    prune_ops.prune_single_block(blk, torch.tensor(keep_h), torch.tensor(keep_m))
                u["mlp"] = [u["mlp"][i] for i in keep_m]
            u["heads"] = [u["heads"][i] for i in keep_h]
            report[f"{kind}:{orig}"] = {"heads": len(u["heads"]),
                                        "ffn": len(u["ff"]) if kind == "double" else len(u["mlp"])}
    return report


def select_blocks_to_skip(block_sim: dict, tau: float, max_blocks: int | None = None,
                          allow_consecutive: bool = False, protect: list | None = None) -> list[int]:
    """sim(B_i, B_{i+1}) > τ 时标记 B_{i+1} 可跳过；按相似度从高到低选取，最多 max_blocks 个。"""
    protect = set(protect or [])
    cands = sorted(((s, b) for (a, b), s in block_sim.items() if s > tau and b not in protect), reverse=True)
    chosen = []
    for s, b in cands:
        if max_blocks is not None and len(chosen) >= max_blocks:
            break
        if not allow_consecutive and any(abs(b - c) == 1 for c in chosen):
            continue
        chosen.append(b)
    return sorted(chosen)


def prune_blocks(model, block_sim: dict, cfg: dict) -> list[int]:
    skip = select_blocks_to_skip(block_sim, cfg.get("tau", 0.92), cfg.get("max_blocks"),
                                 cfg.get("allow_consecutive", False), cfg.get("protect"))
    if skip:
        prune_ops.remove_single_blocks(model, skip)
        units = getattr(model, "ecflux_units", None)
        if units:
            for b in skip:
                units.pop(("single", b), None)
    logger.info(f"block pruning: removed single blocks {skip}")
    return skip

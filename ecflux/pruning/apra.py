"""4.2.2 自适应剪枝率调节机制（APRA）与完整的多粒度压缩流程。

for t in 1..T:
    1. 按 {γ_g} 对 M_{t-1} 执行头 + 神经元剪枝
    2. 快速恢复训练（recover_steps 步，lr=1e-5）
    3. 评估 Q_t（默认 FID，20 步采样）
    4. ΔQ = Q_t - Q_{t-1}；ΔQ < low: γ×1.15；ΔQ > high: γ×0.80；否则 γ×1.02（夹在 [γ_min, γ_max]）
块级剪枝与细粒度剪枝解耦，默认在细粒度剪枝 + 恢复之后执行（block_stage=after）。

消融开关（cfg["apra"]）：
    enabled=false      -> 固定剪枝率一次性剪枝（Fixed-ratio baseline）
    feedback=global    -> 报告原始设定，全局 ΔQ 驱动所有组
    feedback=per_group -> 每组单独探测剪枝敏感度，按组调节（各组差异化）
    rollback=true      -> ΔQ > high 时撤销本轮剪枝
"""

import copy
import logging
import math

import torch
import torch.nn.functional as F

from ..flux_utils import fm_forward
from ..hooks import block_map
from ..utils import build_optimizer, build_scheduler, count_params
from . import prune_ops
from .importance import collect_importance
from .pruner import DEFAULT_GROUPS, init_unit_meta, prune_blocks, prune_to_rates

logger = logging.getLogger("ecflux")


def flow_finetune(model, batch_iter, steps: int, cfg: dict, tracker=None, tag="recover"):
    """剪枝后的快速恢复训练：标准 Rectified Flow 目标 ||v_θ(x_t,t,c) - (ε - x0)||²。"""
    if steps <= 0:
        return
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    opt = build_optimizer(model.parameters(), cfg)
    sched = build_scheduler(opt, steps, cfg.get("warmup_steps", 0))
    accum = int(cfg.get("grad_accum", 1))
    dev_type = next(model.parameters()).device.type
    amp_dtype = torch.bfloat16 if cfg.get("mixed_precision", "bf16") == "bf16" else torch.float16
    use_amp = dev_type == "cuda" and cfg.get("mixed_precision", "bf16") != "no"
    for step in range(steps):
        for _ in range(accum):
            fb = next(batch_iter)
            with torch.autocast(dev_type, dtype=amp_dtype, enabled=use_amp):
                loss = F.mse_loss(fm_forward(model, fb).float(), fb["target"].float()) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if tracker and (step % cfg.get("log_every", 50) == 0 or step == steps - 1):
            tracker.log({f"{tag}/loss": loss.item() * accum, f"{tag}/lr": sched.get_last_lr()[0]})
    if cfg.get("gradient_checkpointing", True):
        model.disable_gradient_checkpointing()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del opt


def _update_rate(rate, delta, acfg, gmin, gmax):
    if delta < acfg.get("low", 0.5):
        rate = rate * acfg.get("up", 1.15)
    elif delta > acfg.get("high", 2.0):
        rate = rate * acfg.get("down", 0.80)
    else:
        rate = rate * acfg.get("mid", 1.02)
    return float(min(max(rate, gmin), gmax))


def _probe_group(model, importance, rates, groups, g, pcfg, evaluator, base_score):
    """只对组 g 的块按候选剪枝率临时剪枝（深拷贝这些块），测量质量退化后恢复原块。"""
    bm = block_map(model)
    grp = next(x for x in groups if x["name"] == g)
    saved, saved_units = [], {}
    units = init_unit_meta(model)
    for kind in ("double", "single"):
        blocks = model.transformer_blocks if kind == "double" else model.single_transformer_blocks
        for orig in grp.get(kind, []):
            if orig in bm[kind]:
                pos = bm[kind].index(orig)
                saved.append((blocks, pos, blocks[pos]))
                blocks[pos] = copy.deepcopy(blocks[pos])
                saved_units[(kind, orig)] = copy.deepcopy(units[(kind, orig)])
    try:
        prune_to_rates(model, importance, rates, [grp], pcfg)
        score = evaluator(model)
    finally:
        for blocks, pos, mod in saved:
            blocks[pos] = mod
        units.update(saved_units)
    return score - base_score


def compress(model, calib_batches, recover_iter, evaluator, cfg: dict, tracker=None) -> dict:
    """完整压缩流程（细粒度 APRA + 块级剪枝）。原地修改 model，返回过程日志。"""
    pcfg, acfg, bcfg, rcfg = cfg["prune"], cfg.get("apra", {}), cfg.get("block", {}), cfg.get("recover", {})
    groups = cfg.get("groups") or DEFAULT_GROUPS
    gmin_default, gmax_default = acfg.get("gamma_min", 0.02), acfg.get("gamma_max", 0.5)
    uniform = acfg.get("uniform_rate")  # 消融：所有组使用同一剪枝率
    rates = {g["name"]: float(uniform if uniform is not None else g["init"]) for g in groups}
    bounds = {g["name"]: (g.get("min", gmin_default), g.get("max", gmax_default)) for g in groups}
    n0 = count_params(model)
    history = {"params_orig": n0, "rounds": []}
    init_unit_meta(model)

    block_stage = bcfg.get("stage", "after") if pcfg.get("prune_blocks", True) else "none"

    def do_block_stage():
        imp = collect_importance(model, calib_batches, {**pcfg, "head_metric": "magnitude",
                                                         "neuron_metric": "magnitude", "modality_aware": False})
        removed = prune_blocks(model, imp["block_sim"], bcfg)
        flow_finetune(model, recover_iter, rcfg.get("block_steps", rcfg.get("steps", 2000)), rcfg, tracker, "block")
        history["removed_single_blocks"] = removed

    if block_stage == "before":
        do_block_stage()

    score_prev = evaluator(model)
    history["score_0"] = score_prev
    logger.info(f"[APRA] initial score={score_prev:.4f} params={n0 / 1e6:.2f}M")
    rounds = int(acfg.get("rounds", 5)) if acfg.get("enabled", True) else 1
    target = acfg.get("target_param_ratio")  # 例如 0.7：参数量降到原始的 70% 即停止
    for t in range(1, rounds + 1):
        if pcfg.get("prune_heads", True) or pcfg.get("prune_neurons", True):
            importance = collect_importance(model, calib_batches, pcfg, seed=t)
        else:
            importance = None
        if acfg.get("enabled", True) and acfg.get("feedback", "global") == "per_group" and importance is not None:
            for g in groups:
                cand = dict(rates)
                cand[g["name"]] = min(rates[g["name"]] * acfg.get("up", 1.15), bounds[g["name"]][1])
                d = _probe_group(model, importance, cand, groups, g["name"], pcfg, evaluator, score_prev)
                pg = acfg.get("per_group_thresholds", {})
                rates[g["name"]] = _update_rate(rates[g["name"]], d, {**acfg, **pg}, *bounds[g["name"]])
                logger.info(f"[APRA] probe {g['name']}: Δ={d:.4f} -> γ={rates[g['name']]:.4f}")

        snapshot = None
        if acfg.get("rollback", False):
            snapshot = _snapshot(model)
        layer_report = prune_to_rates(model, importance, rates, groups, pcfg) if importance else {}
        flow_finetune(model, recover_iter, rcfg.get("steps", 2000), rcfg, tracker, f"recover_r{t}")
        score = evaluator(model)
        delta = score - score_prev
        n = count_params(model)
        rec = {"round": t, "rates": dict(rates), "score": score, "delta": delta, "params": n,
               "param_ratio": n / n0, "layers": layer_report}
        logger.info(f"[APRA] round {t}: score={score:.4f} Δ={delta:+.4f} params={n / 1e6:.2f}M ({n / n0:.3f})")
        if tracker:
            tracker.log({"apra/score": score, "apra/delta": delta, "apra/param_ratio": n / n0,
                         **{f"apra/gamma_{k}": v for k, v in rates.items()}}, step=t)

        if acfg.get("enabled", True) and acfg.get("feedback", "global") == "global":
            rates = {g: _update_rate(r, delta, acfg, *bounds[g]) for g, r in rates.items()}
        rejected = snapshot is not None and delta > acfg.get("high", 2.0)
        if rejected:
            _restore(model, snapshot)
            logger.info(f"[APRA] round {t} rejected (Δ>{acfg.get('high', 2.0)}), rolled back")
        else:
            score_prev = score
        rec["rejected"] = rejected
        history["rounds"].append(rec)

        if target is not None and count_params(model) / n0 <= target:
            logger.info("[APRA] reached target compression ratio")
            break
        if all(math.isclose(r, bounds[g][1]) or math.isclose(r, bounds[g][0]) for g, r in rates.items()):
            logger.info("[APRA] all group rates hit bounds, converged")
            break

    if block_stage == "after":
        do_block_stage()
    history["final_score"] = evaluator(model)
    history["params_final"] = count_params(model)
    history["final_rates"] = rates
    history["structure"] = prune_ops.structure_spec(model)
    return history


def _snapshot(model) -> dict:
    """逐块拷贝到 CPU，避免整模型 deepcopy 带来的显存峰值。"""
    return {
        "double": [copy.deepcopy(b).cpu() for b in model.transformer_blocks],
        "single": [copy.deepcopy(b).cpu() for b in model.single_transformer_blocks],
        "others": {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                   if not n.startswith(("transformer_blocks.", "single_transformer_blocks."))},
        "block_map": copy.deepcopy(block_map(model)),
        "units": copy.deepcopy(model.ecflux_units),
    }


def _restore(model, snap: dict):
    dev = next(model.parameters()).device
    model.transformer_blocks = torch.nn.ModuleList([b.to(dev) for b in snap["double"]])
    model.single_transformer_blocks = torch.nn.ModuleList([b.to(dev) for b in snap["single"]])
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in snap["others"].items():
            own[n].copy_(v.to(dev))
    model.ecflux_block_map = snap["block_map"]
    model.ecflux_units = snap["units"]

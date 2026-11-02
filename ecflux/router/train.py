"""判别器训练（BCE，参考 Hybrid LLM / RouteLLM）与云端预算约束下的阈值校准。"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from .model import ComplexityRouter

logger = logging.getLogger("ecflux")


def train_router(model: ComplexityRouter, x_tr, y_tr, x_va, y_va, cfg: dict):
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("wd", 1e-2)))
    pos = y_tr.float().mean().clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - pos) / pos) if cfg.get("balance", True) else torch.tensor(1.0)
    bs, epochs = int(cfg.get("batch_size", 128)), int(cfg.get("epochs", 30))
    best, best_state = -1.0, None
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(model(x_tr[idx]), y_tr[idx].float(), pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        p = model.prob(x_va).numpy()
        auc = roc_auc(y_va.numpy(), p)
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            logger.info(f"[router] epoch {ep} loss={loss.item():.4f} val_auc={auc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def classification_report(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"tau": float(tau), "accuracy": float((pred == y).mean()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-8, prec + rec), "specificity": tn / max(1, tn + fp),
            "cloud_rate": float(pred.mean())}


def tradeoff_curve(p: np.ndarray, y: np.ndarray, q_edge: np.ndarray | None = None, q_cloud: np.ndarray | None = None,
                   n_points: int = 51) -> list[dict]:
    """成本-质量权衡曲线：每个阈值下的云端调用率、召回率与（若提供逐 prompt 质量）系统平均质量。"""
    out = []
    for tau in np.linspace(0, 1, n_points):
        r = classification_report(y, p, tau)
        if q_edge is not None and q_cloud is not None:
            r["system_quality"] = float(np.where(p >= tau, q_cloud, q_edge).mean())
        out.append(r)
    return out


def calibrate_threshold(p: np.ndarray, y: np.ndarray, cloud_budget: float | None = None, objective: str = "f1") -> float:
    """cloud_budget 给定时：取满足 云端调用率 ≤ budget 的最小阈值（尽量多地把复杂请求送上云）；
    否则在验证集上最大化 objective（f1 / accuracy / youden）。"""
    if cloud_budget is not None:
        if cloud_budget <= 0:
            return 1.0 + 1e-6
        # 在候选阈值（各样本得分）中取满足预算的最小值；考虑并列得分，保证调用率不超预算
        for tau in np.sort(np.unique(p)):
            if (p >= tau).mean() <= cloud_budget + 1e-12:
                return float(tau)
        return float(p.max() + 1e-6)
    best_tau, best = 0.5, -1
    for tau in np.unique(np.concatenate([p, [0.5]])):
        r = classification_report(y, p, tau)
        score = r["recall"] + r["specificity"] - 1 if objective == "youden" else r[objective]
        if score > best:
            best, best_tau = score, float(tau)
    return best_tau

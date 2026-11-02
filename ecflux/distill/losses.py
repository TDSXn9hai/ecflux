"""4.3.2 三级蒸馏损失：L_total = λ_vel·L_vel + λ_hid·L_hid + λ_attn·L_attn

(1) 速度场对齐   L_vel = E||v_S(x_t,t,c) - v_T(x_t,t,c)||²；robust 版本对学生输入加 δ~N(0, σ²I)，σ=0.01
(2) 隐藏层对齐   L_hid = Σ_l (1 - cos(Proj_S(h_S^l), h_T^l))，Proj_S 为逐层线性投影（1×1 conv 等价）
(3) 注意力对齐   L_attn = Σ_{l∈前k个Double Block} ||A_S^l - A_T^l||_F²，A 为图像 query → 文本 key 的注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def velocity_loss(v_s: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_s.float(), v_t.float())


def hidden_loss(h_s: torch.Tensor, h_t: torch.Tensor, kind: str = "cos") -> torch.Tensor:
    """h_*: (B, N, D)。cos：逐 token 余弦距离均值；mse：消融用的 L2 版本（先做 LayerNorm 消除尺度）。"""
    h_s, h_t = h_s.float(), h_t.float()
    if kind == "cos":
        return (1.0 - F.cosine_similarity(h_s, h_t, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(F.layer_norm(h_s, h_s.shape[-1:]), F.layer_norm(h_t, h_t.shape[-1:]))
    raise ValueError(kind)


def attention_loss(a_s: torch.Tensor, a_t: torch.Tensor, head_mode: str = "mean", head_index=None,
                   reduction: str = "mean", kind: str = "fro") -> torch.Tensor:
    """a_*: (B, H, Nq, T)。

    head_mode:
      mean  —— 师生头数不同（4B 剪枝学生 vs 9B 教师）时对头取平均后比较
      match —— 头一一对应；head_index 给出学生每个保留头对应的教师头下标（教师为原始 4B 时可用）
    kind: fro（报告原式，Frobenius²）或 kl（消融：按文本维 KL）
    """
    a_s, a_t = a_s.float(), a_t.float()
    t = min(a_s.shape[-1], a_t.shape[-1])
    a_s, a_t = a_s[..., :t], a_t[..., :t]
    if a_s.shape[2] != a_t.shape[2]:  # query 下采样步长不同时，对齐到较少的一方
        n = min(a_s.shape[2], a_t.shape[2])
        a_s = a_s[:, :, torch.linspace(0, a_s.shape[2] - 1, n).long()]
        a_t = a_t[:, :, torch.linspace(0, a_t.shape[2] - 1, n).long()]
    if head_mode == "mean":
        a_s, a_t = a_s.mean(1, keepdim=True), a_t.mean(1, keepdim=True)
    elif head_mode == "match":
        if head_index is not None:
            a_t = a_t[:, torch.as_tensor(head_index, device=a_t.device)]
        if a_s.shape[1] != a_t.shape[1]:
            raise ValueError(f"head_mode=match but heads differ: {a_s.shape[1]} vs {a_t.shape[1]}")
    else:
        raise ValueError(head_mode)
    if kind == "kl":
        p_t = a_t / a_t.sum(-1, keepdim=True).clamp_min(1e-8)
        p_s = a_s / a_s.sum(-1, keepdim=True).clamp_min(1e-8)
        return (p_t * (p_t.clamp_min(1e-8).log() - p_s.clamp_min(1e-8).log())).sum(-1).mean()
    diff = (a_s - a_t).pow(2)
    # Frobenius² 按矩阵求和；mean 归一化到每个元素，便于在不同分辨率 / 序列长度下复用 λ
    return diff.sum(dim=(-1, -2)).mean() if reduction == "sum" else diff.mean() * diff.shape[-1]


class HiddenProjectors(nn.Module):
    """学生隐藏状态到教师维度的逐层线性投影 Proj_S。维度相同时以单位阵初始化。"""

    def __init__(self, names: list[str], d_student: int, d_teacher: int):
        super().__init__()
        self.names = list(names)
        self.proj = nn.ModuleDict()
        for n in self.names:
            lin = nn.Linear(d_student, d_teacher, bias=False)
            if d_student == d_teacher:
                nn.init.eye_(lin.weight)
            else:
                nn.init.normal_(lin.weight, std=d_student ** -0.5)
            self.proj[n.replace(":", "_")] = lin

    def forward(self, name: str, h: torch.Tensor) -> torch.Tensor:
        lin = self.proj[name.replace(":", "_")]
        return lin(h.to(lin.weight.dtype))


def ramp(step: int, start: int, end: int) -> float:
    """递进式权重：start 之前为 0，start→end 线性升到 1。"""
    if end <= start:
        return 1.0 if step >= start else 0.0
    return float(min(1.0, max(0.0, (step - start) / (end - start))))


class DistillLoss(nn.Module):
    def __init__(self, cfg: dict, projectors: HiddenProjectors | None, student_hidden: list[str],
                 teacher_hidden: list[str], attn_layers: list[int]):
        super().__init__()
        self.cfg = cfg
        self.projectors = projectors
        self.s_hidden, self.t_hidden = student_hidden, teacher_hidden
        self.attn_layers = attn_layers

    def weights(self, step: int) -> dict:
        c = self.cfg
        w = {"vel": c.get("lambda_vel", 1.0) if c.get("use_vel", True) else 0.0,
             "hid": c.get("lambda_hid", 0.5) if c.get("use_hid", True) else 0.0,
             "attn": c.get("lambda_attn", 0.1) if c.get("use_attn", True) else 0.0,
             "gt": c.get("lambda_gt", 0.0)}
        if c.get("progressive", True):
            r = ramp(step, c.get("ramp_start", 1000), c.get("ramp_end", 5000))
            w["hid"] *= r
            w["attn"] *= r
        return w

    def forward(self, step, v_s, v_t, rec_s, teacher_hidden: dict, teacher_attn: dict, target=None,
                head_index: dict | None = None):
        w = self.weights(step)
        logs, total = {}, 0.0
        if w["vel"] > 0:
            l_vel = velocity_loss(v_s, v_t)
            total = total + w["vel"] * l_vel
            logs["loss_vel"] = l_vel.item()
        if w["gt"] > 0 and target is not None:
            l_gt = velocity_loss(v_s, target)
            total = total + w["gt"] * l_gt
            logs["loss_gt"] = l_gt.item()
        if w["hid"] > 0 and self.s_hidden:
            l_hid = 0.0
            for sn, tn in zip(self.s_hidden, self.t_hidden):
                h_s = self.projectors(sn, rec_s.hidden[sn])
                l_hid = l_hid + hidden_loss(h_s, teacher_hidden[tn].to(h_s.device), self.cfg.get("hidden_kind", "cos"))
            total = total + w["hid"] * l_hid
            logs["loss_hid"] = float(l_hid.detach()) if torch.is_tensor(l_hid) else l_hid
        if w["attn"] > 0 and self.attn_layers:
            l_attn = 0.0
            for l in self.attn_layers:
                if l not in rec_s.attn or l not in teacher_attn:
                    continue
                l_attn = l_attn + attention_loss(
                    rec_s.attn[l], teacher_attn[l].to(rec_s.attn[l].device), self.cfg.get("attn_head_mode", "mean"),
                    (head_index or {}).get(l), self.cfg.get("attn_reduction", "mean"), self.cfg.get("attn_kind", "fro"))
            total = total + w["attn"] * l_attn
            logs["loss_attn"] = float(l_attn.detach()) if torch.is_tensor(l_attn) else l_attn
        if not torch.is_tensor(total):  # 所有项权重为 0（如消融 + 递进权重尚未升起）时保持计算图有效
            total = v_s.float().sum() * 0.0
        logs.update({f"w_{k}": v for k, v in w.items()})
        return total, logs

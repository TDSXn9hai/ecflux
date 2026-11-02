"""4.4.1 复杂度标签构造：以「云端教师 - 端侧学生」的综合质量差距作为信号。

输入记录（每条 prompt）：
    {"prompt": ..., "clip_edge", "clip_cloud", "hps_edge", "hps_cloud", ["geneval_edge", "geneval_cloud"]}
gap = Σ_m w_m · norm(q_cloud^m - q_edge^m)；gap > δ 记为复杂(1)，否则简单(0)。
norm=zscore 时各指标差值按全体样本标准化，避免量纲差异（CLIP≈30、HPS≈0.3、GenEval∈{0,1}）。
"""

import numpy as np

METRICS = ("clip", "hps", "geneval")


def quality_gap(records: list[dict], weights: dict | None = None, norm: str = "zscore") -> np.ndarray:
    weights = weights or {"clip": 1.0, "hps": 1.0, "geneval": 1.0}
    gap = np.zeros(len(records))
    for m in METRICS:
        w = weights.get(m, 0.0)
        if w == 0 or not all(f"{m}_edge" in r and f"{m}_cloud" in r for r in records):
            continue
        d = np.array([r[f"{m}_cloud"] - r[f"{m}_edge"] for r in records], dtype=np.float64)
        if norm == "zscore":
            d = (d - d.mean()) / (d.std() + 1e-8)
        gap += w * d
    return gap


def make_labels(records: list[dict], weights=None, norm="zscore", threshold: float = 0.0,
                threshold_mode: str = "value") -> tuple[np.ndarray, np.ndarray]:
    """threshold_mode=value：gap > threshold；quantile：最差的 (1-threshold) 分位标为复杂。"""
    gap = quality_gap(records, weights, norm)
    thr = np.quantile(gap, threshold) if threshold_mode == "quantile" else threshold
    return (gap > thr).astype(np.int64), gap

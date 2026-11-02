"""4.4.2 端云协同推理。

请求 -> 判别器得到 p(复杂)
    p <  τ - band          : 端侧学生直接生成
    p >= τ + band          : 云端教师生成
    |p - τ| < band 且 cascade : 端侧先生成，HPSv2 预估分 < hps_gate 时再触发云端重生成（FrugalGPT 式级联）
对比基线：mode = edge（全端侧）/ cloud（全云端）/ random（固定比例随机路由）/ router（本方法）
"""

import logging
import random
import time
from dataclasses import dataclass, field

import torch

from ..flux_utils import masked_mean_pool
from .features import struct_feature_tensor

logger = logging.getLogger("ecflux")


@dataclass
class RouteResult:
    prompt: str
    image: object
    route: str  # edge / cloud / cascade_edge / cascade_cloud
    prob: float | None
    latency_s: float
    cloud_called: bool
    hps_edge: float | None = None
    extra: dict = field(default_factory=dict)


class EdgeCloudSystem:
    """edge_fn / cloud_fn: (prompt, seed, prompt_embeds|None) -> PIL.Image；cloud_fn 可以是本地教师或远端服务客户端。"""

    def __init__(self, edge_fn, cloud_fn, router=None, text_encoder=None, tau: float = 0.5, band: float = 0.0,
                 cascade: bool = False, hps_scorer=None, hps_gate: float = 0.25, mode: str = "router",
                 random_ratio: float = 0.3, seed: int = 0):
        self.edge_fn, self.cloud_fn = edge_fn, cloud_fn
        self.router, self.text_encoder = router, text_encoder
        self.tau, self.band, self.cascade = tau, band, cascade
        self.hps_scorer, self.hps_gate = hps_scorer, hps_gate
        self.mode, self.random_ratio = mode, random_ratio
        self.rng = random.Random(seed)
        if cascade and hps_scorer is None:
            raise ValueError("cascade=True requires an hps_scorer")

    @torch.no_grad()
    def complexity(self, prompt: str):
        """返回 (p, prompt_embeds)。prompt_embeds 可直接复用给端侧生成，判别器几乎零额外开销。"""
        emb, mask = self.text_encoder.encode([prompt], return_mask=True)
        x = self.router.build_input(masked_mean_pool(emb, mask).cpu(), struct_feature_tensor([prompt]))
        return float(self.router.prob(x)[0]), emb

    def __call__(self, prompt: str, seed: int = 0) -> RouteResult:
        t0 = time.perf_counter()
        p, emb = None, None
        if self.mode == "edge":
            decision = "edge"
        elif self.mode == "cloud":
            decision = "cloud"
        elif self.mode == "random":
            decision = "cloud" if self.rng.random() < self.random_ratio else "edge"
        elif self.mode == "router":
            p, emb = self.complexity(prompt)
            if self.cascade and abs(p - self.tau) < self.band:
                decision = "cascade"
            else:
                decision = "cloud" if p >= self.tau else "edge"
        else:
            raise ValueError(self.mode)

        hps = None
        if decision == "edge":
            img, route, cloud = self.edge_fn(prompt, seed, emb), "edge", False
        elif decision == "cloud":
            img, route, cloud = self.cloud_fn(prompt, seed, None), "cloud", True
        else:
            img = self.edge_fn(prompt, seed, emb)
            hps = float(self.hps_scorer([img], [prompt])[0])
            if hps < self.hps_gate:
                img, route, cloud = self.cloud_fn(prompt, seed, None), "cascade_cloud", True
            else:
                route, cloud = "cascade_edge", False
        return RouteResult(prompt, img, route, p, time.perf_counter() - t0, cloud, hps)


def summarize(results: list[RouteResult]) -> dict:
    n = max(1, len(results))
    routes = {}
    for r in results:
        routes[r.route] = routes.get(r.route, 0) + 1
    lat = sorted(r.latency_s for r in results)
    return {"n": len(results), "cloud_call_rate": sum(r.cloud_called for r in results) / n,
            "avg_latency_s": sum(lat) / n, "p95_latency_s": lat[int(0.95 * (len(lat) - 1))] if lat else 0.0,
            "routes": routes}

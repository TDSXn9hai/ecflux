"""通用工具：配置加载、随机种子、日志、W&B 封装、优化器构造。"""

import copy
import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger("ecflux")


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- config


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_by_path(cfg: dict, dotted: str, value: Any):
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_config(path: str | None, overrides: list[str] | None = None, defaults: dict | None = None) -> dict:
    """读取 YAML 配置，支持 `_base_` 继承与 `a.b.c=value` 形式的命令行覆盖（value 按 YAML 解析）。"""
    cfg = copy.deepcopy(defaults) if defaults else {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        base = file_cfg.pop("_base_", None)
        if base:
            base_path = os.path.join(os.path.dirname(path), base)
            cfg = _deep_update(cfg, load_config(base_path))
        cfg = _deep_update(cfg, file_cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_by_path(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_prompts(path: str) -> list[str]:
    """支持 .txt（每行一条）与 .jsonl（字段 prompt / caption）。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                d = json.loads(line)
                out.append(d.get("prompt") or d.get("caption"))
            else:
                out.append(line)
    return out


# ----------------------------------------------------------------------------- torch helpers


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_dtype(name: str) -> torch.dtype:
    return DTYPES[name]


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_optimizer(params, cfg: dict):
    """AdamW 8-bit（bitsandbytes）优先，不可用或在 CPU 上时回退到 torch.optim.AdamW。"""
    lr = float(cfg.get("lr", 1e-5))
    wd = float(cfg.get("weight_decay", 1e-2))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    params = [p for p in params if p.requires_grad]
    if cfg.get("use_8bit_adam", True) and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd, betas=betas)
        except ImportError:
            logger.warning("bitsandbytes 不可用，回退到 torch.optim.AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
    """线性 warmup + 余弦退火。"""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Tracker:
    """W&B 的薄封装；未安装或 disabled 时退化为日志打印。"""

    def __init__(self, cfg: dict | None, run_config: dict | None = None):
        cfg = cfg or {}
        self.run = None
        if cfg.get("enabled", False):
            try:
                import wandb

                self.run = wandb.init(project=cfg.get("project", "ecflux"), name=cfg.get("name"), config=run_config)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"wandb 初始化失败（{e}），仅打印日志")

    def log(self, metrics: dict, step: int | None = None):
        if self.run is not None:
            self.run.log(metrics, step=step)
        else:
            logger.info(f"step={step} " + " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

    def finish(self):
        if self.run is not None:
            self.run.finish()


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**3
    return float("nan")

"""4.4.1 任务复杂度判别器：冻结 Text Encoder 嵌入（masked mean pool）[+ 结构特征] -> 两层 MLP -> P(复杂)。"""

import torch
import torch.nn as nn

from .features import FEATURE_NAMES


class ComplexityRouter(nn.Module):
    def __init__(self, emb_dim: int, features: str = "emb", hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.features, self.emb_dim = features, emb_dim
        in_dim = (emb_dim if "emb" in features else 0) + (len(FEATURE_NAMES) if "struct" in features else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.config = {"emb_dim": emb_dim, "features": features, "hidden": hidden, "dropout": dropout}

    def build_input(self, pooled_emb: torch.Tensor | None, struct: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if "emb" in self.features:
            parts.append(pooled_emb.float())
        if "struct" in self.features:
            parts.append(struct.float())
        return torch.cat(parts, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x 为 build_input 的输出；返回 logits (B,)。"""
        return self.mlp(self.norm(x)).squeeze(-1)

    @torch.no_grad()
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def save(self, path: str, extra: dict | None = None):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        m.extra = ck.get("extra", {})
        return m.eval()

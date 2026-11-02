"""真实 FLUX.2 Klein 4B 权重上的 CPU 结构检查（约需 10GB 内存，不需要 GPU / 文本编码器）。

python tests/real_weights_check.py weights/FLUX.2-klein-base-4B
验证：权重加载、三粒度重要性统计、按表1 剪枝率物理剪枝、块剪枝、结构 spec 重建形状一致。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ecflux.flux_utils import add_noise, load_transformer, prepare_latent_ids, prepare_text_ids  # noqa: E402
from ecflux.pruning import prune_ops  # noqa: E402
from ecflux.pruning.importance import collect_importance  # noqa: E402
from ecflux.pruning.pruner import DEFAULT_GROUPS, init_unit_meta, prune_blocks, prune_to_rates  # noqa: E402
from ecflux.utils import count_params  # noqa: E402


def main(path):
    t0 = time.time()
    m = load_transformer(path, dtype=torch.bfloat16).eval().requires_grad_(False)
    c = m.config
    print(f"loaded in {time.time() - t0:.0f}s: params={count_params(m) / 1e9:.3f}B double={len(m.transformer_blocks)} "
          f"single={len(m.single_transformer_blocks)} heads={c.num_attention_heads} head_dim={c.attention_head_dim} "
          f"joint_dim={c.joint_attention_dim}")
    torch.manual_seed(0)
    lat = torch.randn(1, c.in_channels, 4, 4)
    emb = torch.randn(1, 16, c.joint_attention_dim, dtype=torch.bfloat16)
    x0 = lat.flatten(2).transpose(1, 2)
    t = torch.tensor([0.6])
    x_t, v = add_noise(x0, t, torch.randn_like(x0))
    fb = {"x_t": x_t.bfloat16(), "t": t.bfloat16(), "target": v, "prompt_embeds": emb,
          "img_ids": prepare_latent_ids(lat), "txt_ids": prepare_text_ids(emb)}
    imp = collect_importance(m, [fb], {"head_metric": "grad", "neuron_metric": "activation", "modality_aware": True})
    print("head grad score (double0):", imp["heads"][("double", 0)][:6].tolist())
    print("head text ratio (double0):", [round(float(x), 3) for x in imp["head_text_ratio"][("double", 0)][:6]])
    print("max block sim:", max(imp["block_sim"].items(), key=lambda kv: kv[1]))
    n0 = count_params(m)
    init_unit_meta(m)
    rates = {g["name"]: g["init"] for g in DEFAULT_GROUPS}
    prune_to_rates(m, imp, rates, DEFAULT_GROUPS, {"alpha_cross": 0.15, "alpha_self": 0.40, "beta_ffn": 0.35,
                                                   "cross_ratio_threshold": 1.0})
    prune_blocks(m, imp["block_sim"], {"tau": -1.0, "max_blocks": 2})
    n1 = count_params(m)
    with torch.no_grad():
        out = m(hidden_states=fb["x_t"], timestep=fb["t"], encoder_hidden_states=emb, img_ids=fb["img_ids"],
                txt_ids=fb["txt_ids"], return_dict=False)[0]
    print(f"pruned: {n0 / 1e9:.3f}B -> {n1 / 1e9:.3f}B ({n1 / n0:.3f}); heads per block:",
          [b.attn.heads for b in m.transformer_blocks], [b.attn.heads for b in m.single_transformer_blocks][:8], "...")
    assert out.shape == x0.shape and torch.isfinite(out.float()).all()
    spec = prune_ops.structure_spec(m)
    from diffusers import Flux2Transformer2DModel

    with torch.device("meta"):
        m2 = Flux2Transformer2DModel.from_config(spec["base_config"])
    prune_ops.apply_structure_spec(m2, spec)
    s1 = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    s2 = {k: tuple(v.shape) for k, v in m2.state_dict().items()}
    assert s1 == s2, "spec rebuild mismatch"
    print("structure spec rebuild matches; REAL WEIGHTS CHECK PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights/FLUX.2-klein-base-4B")

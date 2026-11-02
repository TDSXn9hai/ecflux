"""CPU 冒烟测试：用与 FLUX.2 Klein 同拓扑（5 Double + 20 Single）的极小随机模型跑通全部流程。

python tests/smoke_test.py            # 全部
python tests/smoke_test.py prune      # 仅某一部分：ops / prune / distill / sample / router
"""

import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ecflux.data import SyntheticCachedDataset, collate_cached  # noqa: E402
from ecflux.flux_utils import euler_sample, fm_forward, load_transformer, prepare_fm_batch, tiny_transformer  # noqa: E402
from ecflux.utils import count_params, seed_everything, setup_logging  # noqa: E402

DEV = torch.device("cpu")
setup_logging()


def tiny_student():
    return tiny_transformer(heads=8, head_dim=16, joint_dim=32, in_channels=16, seed=0).eval()


def fm_batches(n=2, bs=2, embed_dim=32, seed=0, key="prompt_embeds", teacher_dim=None):
    ds = SyntheticCachedDataset(n=n * bs, embed_dim=embed_dim, teacher_embed_dim=teacher_dim, seed=seed)
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        b = collate_cached([ds[j] for j in range(i * bs, (i + 1) * bs)])
        out.append(prepare_fm_batch(b, DEV, torch.float32, g))
    return out


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [ok] {msg}")


# ----------------------------------------------------------------------------- 1. 物理剪枝算子
def test_ops():
    print("== ops")
    from ecflux.pruning import prune_ops

    m = tiny_student()
    fb = fm_batches(1)[0]
    with torch.no_grad():
        ref = fm_forward(m, fb)
    # 1) 「零化某头 / 某神经元的输出列」与「物理删除」必须数值等价
    m_zero, m_cut = copy.deepcopy(m), copy.deepcopy(m)
    hd = 16
    with torch.no_grad():
        a = m_zero.transformer_blocks[1].attn
        a.to_out[0].weight[:, 2 * hd:3 * hd] = 0
        a.to_add_out.weight[:, 2 * hd:3 * hd] = 0
        m_zero.transformer_blocks[1].ff.linear_out.weight[:, 5] = 0
        s = m_zero.single_transformer_blocks[3].attn
        s.to_out.weight[:, 1 * hd:2 * hd] = 0
        s.to_out.weight[:, s.inner_dim + 7] = 0
    keep_h = torch.tensor([h for h in range(8) if h != 2])
    prune_ops.prune_double_heads(m_cut.transformer_blocks[1], keep_h)
    prune_ops.prune_swiglu_ffn(m_cut.transformer_blocks[1].ff, torch.tensor([j for j in range(384) if j != 5]))
    prune_ops.prune_single_block(m_cut.single_transformer_blocks[3], torch.tensor([h for h in range(8) if h != 1]),
                                 torch.tensor([j for j in range(384) if j != 7]))
    with torch.no_grad():
        o_zero, o_cut = fm_forward(m_zero, fb), fm_forward(m_cut, fb)
    check(torch.allclose(o_zero, o_cut, atol=1e-5), "physical head/neuron removal == zero-masking")
    check(count_params(m_cut) < count_params(m), "param count decreases after pruning")
    # 2) 删除 Single Block
    prune_ops.remove_single_blocks(m_cut, [5, 6])
    check(len(m_cut.single_transformer_blocks) == 18 and 5 not in m_cut.ecflux_block_map["single"], "block removal")
    with torch.no_grad():
        out = fm_forward(m_cut, fb)
    check(out.shape == ref.shape, "pruned model forward shape")
    # 3) 保存 / 加载
    with tempfile.TemporaryDirectory() as d:
        prune_ops.save_pruned_transformer(m_cut, d)
        m2 = load_transformer(d, dtype=torch.float32)
        with torch.no_grad():
            out2 = fm_forward(m2, fb)
    check(torch.allclose(out, out2, atol=1e-6), "save/load pruned transformer roundtrip")


# ----------------------------------------------------------------------------- 2. 多粒度剪枝 + APRA
def _prune_cfg(**over):
    cfg = {
        "prune": {"head_metric": "grad", "neuron_metric": "activation", "modality_aware": True,
                  "alpha_cross": 0.15, "alpha_self": 0.40, "beta_ffn": 0.35, "cross_ratio_threshold": 1.0,
                  "prune_heads": True, "prune_neurons": True, "prune_blocks": True},
        "apra": {"enabled": True, "rounds": 2, "feedback": "global", "rollback": True, "low": 0.5, "high": 2.0},
        "block": {"stage": "after", "tau": -1.0, "max_blocks": 2},  # tau=-1 强制触发块剪枝以覆盖代码路径
        "recover": {"steps": 2, "lr": 1e-4, "gradient_checkpointing": True, "use_8bit_adam": False, "log_every": 1},
    }
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def test_prune():
    print("== prune / APRA")
    from ecflux.pruning.apra import compress
    from ecflux.pruning.evaluators import build_evaluator
    from ecflux.pruning.importance import collect_importance

    m = tiny_student()
    calib = fm_batches(2, seed=1)
    imp = collect_importance(m, calib, _prune_cfg()["prune"])
    check(imp["heads"][("double", 0)].shape == (8,) and imp["neurons"][("single", 0, "mlp")].shape == (384,),
          "importance shapes (heads / neurons)")
    check(len(imp["block_sim"]) == 19 and len(imp["head_text_ratio"]) == 25, "block similarity & head modality stats")

    def recover_iter():
        g = torch.Generator().manual_seed(2)
        ds = SyntheticCachedDataset(n=4, seed=3)
        while True:
            yield prepare_fm_batch(collate_cached([ds[0], ds[1]]), DEV, torch.float32, g)

    for name, over in {"global+rollback": {}, "per_group": {"apra": {"feedback": "per_group"}},
                       "fixed(no APRA)": {"apra": {"enabled": False}},
                       "uniform+magnitude": {"apra": {"uniform_rate": 0.2},
                                             "prune": {"head_metric": "magnitude", "neuron_metric": "magnitude"}},
                       "no-modality+block-before": {"prune": {"modality_aware": False}, "block": {"stage": "before"}}}.items():
        mm = tiny_student()
        n0 = count_params(mm)
        ev = build_evaluator("velocity_mse", ref_model=tiny_student(), batches=fm_batches(1, seed=4), device=DEV)
        hist = compress(mm, calib, recover_iter(), ev, _prune_cfg(**over))
        with torch.no_grad():
            fm_forward(mm, calib[0])
        check(count_params(mm) < n0, f"compress[{name}] params {n0} -> {count_params(mm)}, "
                                     f"removed blocks {hist.get('removed_single_blocks')}")
        heads_d0 = mm.transformer_blocks[0].attn.heads
        check(heads_d0 >= 8 - int(0.40 * 8), f"compress[{name}] head cap respected (double0 heads={heads_d0})")


# ----------------------------------------------------------------------------- 3. 蒸馏
def _distill_cfg(**over):
    d = {"lambda_vel": 1.0, "lambda_hid": 0.5, "lambda_attn": 0.1, "use_vel": True, "use_hid": True, "use_attn": True,
         "robust": True, "robust_sigma": 0.01, "progressive": True, "ramp_start": 0, "ramp_end": 2, "attn_k": 2,
         "attn_head_mode": "mean", "attn_query_stride": 2, "teacher_hidden_layers": "auto"}
    d.update(over)
    return {"distill": d, "teacher": {"embeds_key": "teacher_prompt_embeds"},
            "train": {"max_steps": 3, "micro_batch_size": 2, "global_batch_size": 4, "lr": 1e-4, "warmup_steps": 1,
                      "gradient_checkpointing": True, "use_8bit_adam": False, "num_workers": 0, "log_every": 1,
                      "save_every": 0}, "seed": 0}


def test_distill():
    print("== distill")
    from ecflux.distill.trainer import OfflineTeacherDataset, teacher_draws, train
    from ecflux.pruning import prune_ops

    # 学生：剪掉若干头/神经元/块；教师：维度、深度、文本嵌入维度都不同（模拟 9B）
    student = tiny_student()
    prune_ops.prune_double_heads(student.transformer_blocks[0], torch.arange(7))
    prune_ops.remove_single_blocks(student, [3])
    teacher = tiny_transformer(num_layers=6, num_single_layers=24, heads=6, head_dim=16, joint_dim=48, seed=1).eval()
    ds = SyntheticCachedDataset(n=4, embed_dim=32, teacher_embed_dim=48)

    with tempfile.TemporaryDirectory() as d:
        dist = train(student, teacher, ds, _distill_cfg(), DEV, d)
        check(os.path.exists(os.path.join(d, "final", "projectors.pt")), "online distill (vel+hid+attn) trains & saves")
        check(dist.t_hidden == ["double:4", "double:5", "single:13"], f"auto teacher layer map {dist.t_hidden}")
        s2 = load_transformer(os.path.join(d, "final"), dtype=torch.float32)
        check(len(s2.single_transformer_blocks) == 19, "distilled student reload keeps pruned structure")

    for name, over in {"no_vel": {"use_vel": False}, "no_hid": {"use_hid": False}, "no_attn": {"use_attn": False},
                       "mse_hidden+kl_attn+no_robust": {"hidden_kind": "mse", "attn_kind": "kl", "robust": False}}.items():
        with tempfile.TemporaryDirectory() as d:
            cfg = _distill_cfg(**over)
            cfg["train"]["max_steps"] = 1
            train(copy.deepcopy(student), teacher, ds, cfg, DEV, d)
            check(True, f"ablation {name} runs")

    # 同构教师 + 逐头匹配（attn_head_mode=match）
    st = tiny_student()
    from ecflux.pruning.pruner import init_unit_meta, prune_to_rates
    from ecflux.pruning.importance import collect_importance
    calib = fm_batches(1)
    imp = collect_importance(st, calib, {"head_metric": "magnitude", "neuron_metric": "magnitude"})
    init_unit_meta(st)
    prune_to_rates(st, imp, {g: 0.3 for g in ["G1", "G2", "G3", "G4", "G5", "G6"]},
                   __import__("ecflux.pruning.pruner", fromlist=["x"]).DEFAULT_GROUPS, {"alpha_cross": 0.3})
    with tempfile.TemporaryDirectory() as d:
        cfg = _distill_cfg(attn_head_mode="match")
        cfg["teacher"]["embeds_key"] = "prompt_embeds"
        cfg["train"]["max_steps"] = 1
        train(st, tiny_student(), SyntheticCachedDataset(n=4), cfg, DEV, d)
        check(True, "same-arch teacher with per-head matched attention alignment")

    # 离线教师缓存
    with tempfile.TemporaryDirectory() as d:
        g = torch.Generator().manual_seed(0)
        for i in range(4):
            s = ds[i]
            s = dict(s, teacher=teacher_draws(teacher, s, 2, ["double:4", "double:5", "single:13"], [0, 1], DEV,
                                             torch.float32, g, embeds_key="teacher_prompt_embeds"))
            torch.save(s, os.path.join(d, f"{i}.pt"))
        off = OfflineTeacherDataset(d)
        cfg = _distill_cfg()
        cfg["teacher"]["config"] = dict(teacher.config)
        with tempfile.TemporaryDirectory() as o:
            train(copy.deepcopy(student), None, off, cfg, DEV, o)
        check(True, "offline teacher-cache distillation")


# ----------------------------------------------------------------------------- 4. 采样
def test_sample():
    print("== sample")
    m = tiny_student()
    emb = torch.randn(2, 8, 32)
    lat, ids = euler_sample(m, emb, 64, 64, num_steps=3, guidance_scale=4.0, negative_embeds=torch.randn(2, 8, 32))
    check(lat.shape == (2, 16, 16) and torch.isfinite(lat).all(), "Euler sampler with CFG (shape B,N,C)")


# ----------------------------------------------------------------------------- 5. 复杂度判别器 + 端云协同
class _FakeTextEncoder:
    """替代 Qwen3：对 prompt 做确定性哈希嵌入。"""

    def __init__(self, dim=32, length=8):
        self.dim, self.length = dim, length

    def encode(self, prompts, return_mask=False):
        embs = []
        for p in prompts:
            g = torch.Generator().manual_seed(abs(hash(p)) % (2**31))
            e = torch.randn(self.length, self.dim, generator=g) + 0.1 * len(p.split())
            embs.append(e)
        e = torch.stack(embs)
        return (e, torch.ones(e.shape[:2])) if return_mask else e


def test_router():
    print("== router / collab")
    import numpy as np

    from ecflux.flux_utils import masked_mean_pool
    from ecflux.router.collab import EdgeCloudSystem, summarize
    from ecflux.router.features import struct_feature_tensor
    from ecflux.router.labels import make_labels
    from ecflux.router.model import ComplexityRouter
    from ecflux.router.train import calibrate_threshold, classification_report, tradeoff_curve, train_router

    simple = ["a cat", "a red apple", "a dog on grass", "a mountain at sunset"]
    hard = ['three red cubes left of two blue spheres, a sign saying "OPEN"', "a wooden chair behind a glass table "
            "and four green cups on top of a metal box", 'five cats and a "HELLO" neon text between two lamps']
    prompts = (simple * 10) + (hard * 10)
    rng = np.random.default_rng(0)
    recs = []
    for p in prompts:
        h = p in hard
        recs.append({"prompt": p, "clip_edge": 30 - 3 * h + rng.normal(0, .3), "clip_cloud": 31 + rng.normal(0, .3),
                     "hps_edge": .27 - .02 * h + rng.normal(0, .002), "hps_cloud": .28 + rng.normal(0, .002)})
    y, gap = make_labels(recs, threshold=0.0)
    check(y[len(simple) * 10:].mean() > 0.9 and y[: len(simple) * 10].mean() < 0.1, "quality-gap labels separate")

    te = _FakeTextEncoder()
    emb, mask = te.encode(prompts, return_mask=True)
    for feats in ("emb", "struct", "emb+struct"):
        r = ComplexityRouter(emb_dim=32, features=feats, hidden=32)
        x = r.build_input(masked_mean_pool(emb, mask), struct_feature_tensor(prompts))
        yt = torch.tensor(y)
        auc = train_router(r, x, yt, x, yt, {"epochs": 30, "lr": 3e-3, "batch_size": 16})
        p = r.prob(x).numpy()
        tau = calibrate_threshold(p, y, cloud_budget=0.5)
        rep = classification_report(y, p, tau)
        check(rep["cloud_rate"] <= 0.5 + 1e-6, f"router[{feats}] auc={auc:.3f} budget-calibrated tau={tau:.3f} {rep}")
    curve = tradeoff_curve(p, y, np.array([r["hps_edge"] for r in recs]), np.array([r["hps_cloud"] for r in recs]), 5)
    check(len(curve) == 5 and "system_quality" in curve[0], "cost-quality trade-off curve")

    with tempfile.TemporaryDirectory() as d:
        r.save(os.path.join(d, "router.pt"))
        r = ComplexityRouter.load(os.path.join(d, "router.pt"))

    edge = lambda prompt, seed, emb: ("edge_img", prompt)  # noqa: E731
    cloud = lambda prompt, seed, emb: ("cloud_img", prompt)  # noqa: E731
    hps = lambda imgs, ps: [0.2 for _ in imgs]  # noqa: E731
    for mode in ("edge", "cloud", "random", "router"):
        sysm = EdgeCloudSystem(edge, cloud, r, te, tau=tau, mode=mode, random_ratio=0.3)
        s = summarize([sysm(p) for p in prompts])
        check(s["n"] == len(prompts), f"collab mode={mode}: {s}")
    sysm = EdgeCloudSystem(edge, cloud, r, te, tau=tau, band=0.5, cascade=True, hps_scorer=hps, hps_gate=0.25)
    s = summarize([sysm(p) for p in prompts])
    check(any(k.startswith("cascade") for k in s["routes"]), f"cascade routing: {s['routes']}")


# ----------------------------------------------------------------------------- 6. 端到端生成（极小 VAE + 假文本编码器）
def test_generate():
    print("== generate (T2IGenerator / VAE encode-decode)")
    from diffusers import AutoencoderKLFlux2

    from ecflux.flux_utils import encode_images
    from ecflux.generation import T2IGenerator

    with tempfile.TemporaryDirectory() as d:
        tiny_student().save_pretrained(os.path.join(d, "transformer"))
        torch.manual_seed(0)
        AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=4, norm_num_groups=4,
                           layers_per_block=1).save_pretrained(os.path.join(d, "vae"))
        gen = T2IGenerator(d, None, "cpu", torch.float32, text_encoder=_FakeTextEncoder())
        imgs, t = gen(["a red cube", "two cats"], height=128, width=128, steps=2, guidance_scale=4.0,
                      return_timing=True)
        check(len(imgs) == 2 and imgs[0].size == (128, 128), f"T2IGenerator produces PIL images, timing={t}")
        lat = encode_images(gen.vae, torch.rand(1, 3, 128, 128) * 2 - 1)
        check(lat.shape == (1, 16, 8, 8), "VAE encode -> patchified normalized latents (matches DiT in_channels)")


TESTS = {"ops": test_ops, "prune": test_prune, "distill": test_distill, "sample": test_sample, "router": test_router,
         "generate": test_generate}

if __name__ == "__main__":
    seed_everything(0)
    names = sys.argv[1:] or list(TESTS)
    for n in names:
        TESTS[n]()
    print("\nSMOKE TEST PASSED:", ", ".join(names))

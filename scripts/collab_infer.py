"""4.4.2 端云协同推理与基线对比（全端侧 / 全云端 / 随机路由 / 判别器路由 [+级联]）。

python scripts/collab_infer.py --config configs/router.yaml --prompts data/test_prompts.txt \
    --edge_transformer outputs/distill/full/final --router outputs/router/default/router.pt --out outputs/collab/router
python scripts/collab_infer.py ... --set collab.mode=random collab.random_ratio=0.3
单卡 24GB 无法同时常驻 4B 学生与 9B 教师时，用 --cloud_device 指定另一张卡或 cpu（仅用于离线实验）。
"""

import json
import os

from _common import base_parser, init

from ecflux.generation import T2IGenerator
from ecflux.router.collab import EdgeCloudSystem, summarize
from ecflux.router.model import ComplexityRouter
from ecflux.utils import get_dtype, read_prompts, save_json


def main():
    p = base_parser(__doc__, "configs/router.yaml")
    p.add_argument("--prompts", required=True)
    p.add_argument("--edge_transformer", required=True)
    p.add_argument("--router", default=None)
    p.add_argument("--tau", type=float, default=None, help="覆盖 router.pt 中校准得到的阈值")
    p.add_argument("--edge_device", default="cuda")
    p.add_argument("--cloud_device", default="cuda")
    args = p.parse_args()
    cfg = init(args)
    out = args.out or "outputs/collab/default"
    c, dtype = cfg["collab"], get_dtype(cfg.get("dtype", "bf16"))
    prompts = read_prompts(args.prompts)

    edge = T2IGenerator(cfg["paths"]["student_pipeline"], args.edge_transformer, args.edge_device, dtype,
                        cfg["data"]["max_sequence_length"]) if c["mode"] != "cloud" else None
    cloud_holder = {}

    def cloud_gen():  # 懒加载：只有真正需要云端时才加载教师
        if "g" not in cloud_holder:
            cloud_holder["g"] = T2IGenerator(cfg["paths"]["teacher_pipeline"], None, args.cloud_device, dtype,
                                             cfg["data"]["max_sequence_length"])
        return cloud_holder["g"]

    kw = dict(height=c["height"], width=c["width"], steps=c["steps"], guidance_scale=c["guidance_scale"])
    edge_fn = lambda pr, seed, emb: edge([pr], seed=seed, prompt_embeds=emb, **kw)[0]  # noqa: E731
    cloud_fn = lambda pr, seed, emb: cloud_gen()([pr], seed=seed, **kw)[0]  # noqa: E731

    router, tau, hps = None, args.tau, None
    if c["mode"] == "router":
        router = ComplexityRouter.load(args.router)
        tau = tau if tau is not None else router.extra.get("tau", 0.5)
    if c.get("cascade") and c["mode"] == "router":
        from ecflux.metrics import HPSv2Scorer

        hps = HPSv2Scorer()
    system = EdgeCloudSystem(edge_fn, cloud_fn, router, edge.text_encoder if edge else None, tau or 0.5,
                             c.get("band", 0.0), bool(c.get("cascade")) and c["mode"] == "router", hps,
                             c.get("hps_gate", 0.25), c["mode"], c.get("random_ratio", 0.3), cfg["seed"])
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    results = []
    with open(os.path.join(out, "routes.jsonl"), "w", encoding="utf-8") as f:
        for i, pr in enumerate(prompts):
            r = system(pr, seed=cfg["seed"])
            path = os.path.join(out, "images", f"{i:05d}_0.png")
            r.image.save(path)
            results.append(r)
            f.write(json.dumps({"idx": i, "k": 0, "prompt": pr, "path": path, "route": r.route, "prob": r.prob,
                                "latency_s": r.latency_s, "hps_edge": r.hps_edge}, ensure_ascii=False) + "\n")
    # 与 generate.py 相同的 index.jsonl，便于直接用 evaluate.py 评估系统级质量
    os.replace(os.path.join(out, "routes.jsonl"), os.path.join(out, "images", "index.jsonl"))
    s = summarize(results)
    save_json(s, os.path.join(out, "summary.json"))
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

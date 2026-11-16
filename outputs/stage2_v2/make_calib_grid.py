"""恢复训练后阈值标定的样例拼图：行 = prompt，列 = 原始 4B / 漂移参照 ckpt_1（不剪枝 + 2000 步恢复）/ 新初值 × s 剪一轮 + 2000 步恢复。"""
import json
import os

from PIL import Image, ImageDraw

R = "outputs/stage2_v2"
cols = [("orig 4B", "outputs/stage1_v2/base/s42/gen"), ("ref ckpt_1 (no prune)", f"{R}/drift_ref/ckpt_1_gen_s")]
for s in ("0.5", "1.0", "1.5", "2.0"):
    d = f"{R}/calib_scale{s}"
    if os.path.exists(f"{d}/history.json"):
        r = json.load(open(f"{d}/history.json"))["rounds"][0]
        a = r["aux"]
        cols.append((f"x{s} p={r['param_ratio']:.3f} traj_ref={a['traj_ref']:.1f} hps_ref={a['hps_ref']:.4f}", f"{d}/gen_s"))
size, pad = 200, 30
for part, idxs in (("a", range(0, 12)), ("b", range(12, 24))):
    W = size * len(cols)
    c = Image.new("RGB", (W, pad + size * len(idxs)), "white")
    d = ImageDraw.Draw(c)
    for ci, (name, _) in enumerate(cols):
        d.text((ci * size + 2, 2), name[:32], fill="black")
        d.text((ci * size + 2, 14), name[32:64], fill="black")
    for ri, i in enumerate(idxs):
        for ci, (_, g) in enumerate(cols):
            p = f"{g}/{i:05d}_0.png"
            if os.path.exists(p):
                c.paste(Image.open(p).convert("RGB").resize((size, size)), (ci * size, pad + ri * size))
    c.save(f"{R}/calib_grid_{part}.jpg", quality=90)
print([n for n, _ in cols])

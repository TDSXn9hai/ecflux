"""LAION-Aesthetics 6.5+ 的替代训练集：i1 Pexels 512px (1M) 上用 LAION 美学预测器 v2 打分，取高分子集。

python data/prepare_pexels_aes.py score  --src data/raw/i1_pexels_512 --out data/raw/i1_pexels_512_scores
python data/prepare_pexels_aes.py export --src data/raw/i1_pexels_512 --scores data/raw/i1_pexels_512_scores \
    --out data/pexels_aes --num 150000
产物：data/pexels_aes/images/*.png + index.jsonl（{"image","caption","aesthetic"}），可直接交给 cache_dataset.py --source。
美学分 = sac+logos+ava1-l14-linearMSE（LAION-Aesthetics V2 同一预测器）作用于 L2 归一化的 OpenAI CLIP ViT-L/14 图像嵌入。
"""

import argparse
import glob
import io
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_records(path):
    """TFRecord：uint64 长度 + uint32 crc + payload + uint32 crc。产出 (偏移, payload)。"""
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            head = f.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            f.read(4)
            yield off, data


def read_record(path, off):
    with open(path, "rb") as f:
        f.seek(off)
        (n,) = struct.unpack("<Q", f.read(12)[:8])
        return f.read(n)


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(b, i)
            yield fn, b[i:i + n]
            i += n
        elif wt == 0:
            v, i = _varint(b, i)
            yield fn, v
        elif wt == 5:
            yield fn, b[i:i + 4]
            i += 4
        elif wt == 1:
            yield fn, b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"wire type {wt}")


def parse_example(b):
    """tf.train.Example -> {name: [bytes, ...]}（只取 bytes_list，够用）。"""
    out = {}
    for fn, feats in _fields(b):  # Example.features
        if fn != 1:
            continue
        for fn2, entry in _fields(feats):  # Features.feature (map entry)
            if fn2 != 1:
                continue
            name, val = None, b""
            for k, v in _fields(entry):
                if k == 1:
                    name = v.decode()
                elif k == 2:
                    val = v
            vals = []
            for k, lst in _fields(val):  # Feature.bytes_list
                if k == 1:
                    vals += [v for k2, v in _fields(lst) if k2 == 1]
            out[name] = vals
    return out


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                                    nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def _decode(args):
    """worker：PNG -> CLIP 预处理张量（224 bicubic resize + center crop + CLIP 归一化）。"""
    from PIL import Image

    off, data = args
    ex = parse_example(data)
    img = Image.open(io.BytesIO(ex["image"][0])).convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    x = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
    x = (x - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
    caps = [c.decode("utf-8", "replace") for c in ex.get("caption", [])]
    return off, caps, x.transpose(2, 0, 1).astype(np.float32)


def score(a):
    import time
    from multiprocessing import Pool

    from transformers import CLIPModel

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(dev).eval()
    mlp = AestheticMLP()
    mlp.load_state_dict(torch.load(os.path.join(ROOT, "weights/aesthetic/sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu"))
    mlp = mlp.to(dev).eval()
    done = set()
    with Pool(a.workers) as pool:
        while True:
            shards = sorted(glob.glob(os.path.join(a.src, "dataset-train.tfrecord-*")))
            todo = [s for s in shards if os.path.basename(s) not in done
                    and not os.path.exists(os.path.join(a.out, os.path.basename(s) + ".jsonl"))]
            if not todo:
                if a.wait and len(glob.glob(os.path.join(a.out, "*.jsonl"))) < a.num_shards:
                    time.sleep(60)
                    continue
                break
            for sh in todo:
                t0, rows, buf = time.time(), [], []

                def flush():
                    x = torch.from_numpy(np.stack([b[2] for b in buf])).to(dev, torch.float16)
                    with torch.no_grad():
                        e = clip.get_image_features(pixel_values=x)
                        e = e.pooler_output if hasattr(e, "pooler_output") else e
                        e = torch.nn.functional.normalize(e.float(), dim=-1)
                        s = mlp(e).squeeze(1).cpu().tolist()
                    rows.extend({"offset": b[0], "aesthetic": round(v, 4), "captions": b[1]} for b, v in zip(buf, s))
                    buf.clear()

                for item in pool.imap(_decode, iter_records(sh), chunksize=16):
                    buf.append(item)
                    if len(buf) == a.batch_size:
                        flush()
                if buf:
                    flush()
                tmp = os.path.join(a.out, os.path.basename(sh) + ".jsonl.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(tmp, tmp[:-4])
                done.add(os.path.basename(sh))
                sc = np.array([r["aesthetic"] for r in rows])
                print(f"{os.path.basename(sh)}: {len(rows)} imgs, mean {sc.mean():.3f}, >=6.5 {(sc >= 6.5).sum()}, "
                      f"{time.time() - t0:.0f}s", flush=True)


def export(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.scores, "*.jsonl"))):
        shard = os.path.basename(p)[:-6]
        with open(p, encoding="utf-8") as f:
            rows += [dict(json.loads(l), shard=shard) for l in f]
    rows = [r for r in rows if any(c.strip() for c in r["captions"])]
    rows.sort(key=lambda r: -r["aesthetic"])
    sel = rows[: a.num]
    print(f"{len(rows)} scored; selected {len(sel)}; aesthetic range {sel[-1]['aesthetic']:.3f} ~ {sel[0]['aesthetic']:.3f}; "
          f">=6.5: {sum(r['aesthetic'] >= 6.5 for r in rows)}")
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    sel.sort(key=lambda r: (r["shard"], r["offset"]))
    with open(os.path.join(a.out, "index.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(sel):
            name = f"images/{i:06d}.png"
            ex = parse_example(read_record(os.path.join(a.src, r["shard"]), r["offset"]))
            with open(os.path.join(a.out, name), "wb") as w:
                w.write(ex["image"][0])
            # 每图 5 条 VLM 长描述；caption 取最短一条，减少 max_sequence_length 截断，其余保留在 captions
            caps = [" ".join(c.split()) for c in r["captions"] if c.strip()]
            f.write(json.dumps({"image": name, "caption": min(caps, key=len), "captions": caps, "aesthetic": r["aesthetic"],
                                "source": f"i1-pexels-512:{r['shard']}@{r['offset']}"}, ensure_ascii=False) + "\n")
    print("exported ->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["score", "export"])
    p.add_argument("--src", default="data/raw/i1_pexels_512")
    p.add_argument("--scores", default="data/raw/i1_pexels_512_scores")
    p.add_argument("--out", default=None)
    p.add_argument("--num", type=int, default=150000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--num_shards", type=int, default=128)
    p.add_argument("--wait", action="store_true", help="边下载边打分：等待新分片出现直到 num_shards 个全部完成")
    a = p.parse_args()
    if a.cmd == "score":
        a.out = a.out or a.scores
        score(a)
    else:
        a.out = a.out or "data/pexels_aes"
        export(a)

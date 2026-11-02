"""提示词结构特征（4.3 拟解决问题：对象数量、属性绑定、空间关系、文本渲染等维度）。

作为文本嵌入之外的可解释补充特征；features=emb / struct / emb+struct 构成判别器输入的消融。
"""

import re

import torch

NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "a couple of": 2, "several": 3, "many": 5, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5}
COLORS = {"red", "green", "blue", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray", "grey",
          "silver", "golden", "红", "绿", "蓝", "黄", "黑", "白", "紫", "橙", "粉", "棕", "灰"}
ATTRS = {"wooden", "metal", "metallic", "glass", "plastic", "fluffy", "striped", "round", "square", "tall", "small",
         "big", "large", "tiny", "shiny", "leather", "furry", "transparent"}
SPATIAL = ["left of", "right of", "on top of", "above", "below", "under", "beneath", "behind", "in front of",
           "next to", "beside", "between", "inside", "near", "左边", "右边", "上面", "下面", "后面", "前面", "旁边", "之间"]
STYLE = {"photorealistic", "realistic", "oil painting", "watercolor", "anime", "sketch", "3d render", "pixel art"}

FEATURE_NAMES = ["n_words", "n_commas", "n_and", "n_numbers", "max_count", "n_colors", "n_attrs", "n_spatial",
                 "has_quote", "quote_len", "n_style", "n_negation"]


def struct_features(prompt: str) -> list[float]:
    p = prompt.lower()
    words = re.findall(r"[a-z']+|[\u4e00-\u9fff]", p)
    nums = [int(x) for x in re.findall(r"\b\d+\b", p)] + [v for k, v in NUM_WORDS.items() if re.search(rf"\b{k}\b", p)]
    quotes = re.findall(r"[\"“”'‘’「」『』](.+?)[\"“”'‘’「」『』]", prompt)
    return [
        len(words) / 50.0,
        p.count(",") / 5.0,
        len(re.findall(r"\band\b|和|与|以及", p)) / 3.0,
        len(nums) / 3.0,
        (max(nums) if nums else 1) / 10.0,
        sum(w in COLORS for w in words) / 3.0,
        sum(w in ATTRS for w in words) / 3.0,
        sum(s in p for s in SPATIAL) / 2.0,
        float(bool(quotes) or "text" in words or "sign" in words or "saying" in words or "写着" in p),
        sum(len(q) for q in quotes) / 20.0,
        sum(s in p for s in STYLE) / 2.0,
        len(re.findall(r"\bno\b|\bwithout\b|\bnot\b|没有|不", p)) / 2.0,
    ]


def struct_feature_tensor(prompts: list[str]) -> torch.Tensor:
    return torch.tensor([struct_features(p) for p in prompts], dtype=torch.float32)

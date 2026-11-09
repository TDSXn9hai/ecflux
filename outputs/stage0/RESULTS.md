# 阶段0 结果说明（2026-11-01）

## 0.1 FID 随机波动（`fidvar/`）
- 4B 在 D_val（1000 条）上，seed 42/43/44 的 FID 分别为 73.47 / 73.34 / 74.65，均值 73.82，**σ=0.73**。
- 用 seed 42 重跑一次，FID 完全相同，说明生成过程确定、可复现。
- 结论：APRA 的下阈值 0.5 小于 σ。建议阶段2按 σ 重新设定阈值，例如 low≈1σ、high≈3σ。
- 1000 张图算出的 FID 偏高（73 对比 COCO-30k 的 22），只适合做相对比较。

## 0.2 评测环境（`.venv-bench`，依赖清单见 `bench_env_freeze.txt`）
- 主要组件：Python 3.10、torch 2.1.2+cu121、mmcv-full 1.7.2（源码编译，sm_80）、mmdet 2.28.2、detectron2 0.6（通过代理从 github 拉源码编译）、transformers 4.30.2、Pillow 9.5、spacy 3.5 + en_core_web_sm。
- 预置的 `weights/benchmark_eval/detectron2.tar` 是 CUDA 11 编译的，在本机不能用。
- 权重通过软链接接入：`~/.cache/clip/` 下放 ViT-L-14/ViT-B-32；`weights/torch_home/hub/checkpoints/` 下放 BLIP-VQA；T2I-CompBench 的 `UniDet_eval/experts/expert_weights/` 下放 UniDet 与 DPT。
- T2I-CompBench 官方 `3_in_1.py` 写死每条 prompt 10 张图。`run_stage0.sh` 改为按相同加权规则、每条 4 张重新计算，官方代码没有改动。

## 0.3 基线结果（见 `baseline_table.md`）
- 9B 相对 4B 的差距：
  - FID：22.35 → 16.66；GenEval：0.759 → 0.811，主要差在 position（45% → 64%）和 color_attr（60% → 66%）。
  - T2I-CompBench 的 color/shape 高约 5 个点。
  - CLIP、HPSv2、T2I-CompBench non-spatial/complex 两者几乎相同。HPSv2 四类平均 0.2835 对 0.2836，photo 类 9B 反而更低。
- 结论：教师的优势集中在空间关系和属性绑定上。蒸馏阶段的 L_attn（绑定级对齐）以及 GenEval position / color_attr 应作为重点观察的指标。HPSv2 在两个模型之间几乎拉不开差距，用它作为端云路由标签的区分度会比较弱。
- 效率：4B DiT 在 512px 下 40 次前向耗时 2.58s，9B 为 5.34s。端到端峰值显存 4B 为 16GB，9B 为 33.5GB。

## 需要调整开题报告的地方
1. **GenEval 没有 "text" 子集**：只有 single_object、two_object、counting、colors、position、color_attr 六类。表2中的"文本渲染准确率"需要换一个测试集，例如 OCR 类基准，或者自建一组含文字的 prompt 再用 OCR 评测。
2. **硬件**：开题写的是单卡 4090，实际所有数据都测自 A800。部署指标最终仍需在 4090 上补测。
3. **训练数据**：开题写的是 LAION-Aesthetics，实际使用的是 `data/pexels_aes`（15 万条）。

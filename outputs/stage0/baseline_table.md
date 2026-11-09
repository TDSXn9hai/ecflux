# 阶段0 基线结果（表2 / 表3 的 4B、9B 两行）

统一设置：512×512，20 步，CFG 4.0，seed 42，bf16，A800-80GB（不是 4090）。GenEval 和 T2I-CompBench 每条 prompt 生成 4 张图。

## 生成质量
| 指标 | 测试集 | FLUX.2 Klein 4B Base | FLUX.2 Klein 9B Base（教师） |
|---|---|---|---|
| FID↓ | COCO-30k（参考图为对应的 3 万张 COCO val2014 原图，中心裁剪） | 22.35 | 16.66 |
| CLIP Score↑ | COCO-30k（ViT-L/14） | 26.88 | 27.15 |
| HPSv2.1↑ anime / concept-art / paintings / photo | HPSv2 benchmark（各 800 条） | 0.2923 / 0.2828 / 0.2828 / 0.2761 | 0.2988 / 0.2836 / 0.2851 / 0.2666 |
| HPSv2.1↑ 四类平均 | | 0.2835 | 0.2836 |
| GenEval Overall↑ | GenEval 553 条 | 0.759 | 0.811 |
| 　single / two_object | | 99.4% / 88.9% | 99.7% / 90.7% |
| 　counting / colors | | 70.0% / 92.0% | 70.9% / 95.5% |
| 　position / color_attr | | 45.3% / 60.0% | 63.8% / 66.0% |
| T2I-CompBench color / shape / texture（BLIP-VQA）↑ | 验证集各约 300 条 | 0.813 / 0.560 / 0.713 | 0.861 / 0.614 / 0.740 |
| T2I-CompBench spatial（UniDet）↑ | | 0.371 | 0.384 |
| T2I-CompBench non-spatial（CLIP）↑ | | 0.3105 | 0.3132 |
| T2I-CompBench complex（3-in-1）↑ | | 0.385 | 0.387 |

## 效率（只计 DiT，20 步 × CFG 2 次 = 40 次前向，batch 1）
| 指标 | 4B | 9B |
|---|---|---|
| DiT 参数量 | 3.876B | 9.079B |
| 40 次前向延迟 @512 / @1024 | 2.58s / 8.29s | 5.34s / 16.68s |
| DiT 峰值显存 @512 / @1024 | 7.4GB / 7.8GB | 17.2GB / 17.7GB |
| 端到端单图（文本编码 + 采样 + VAE）@512 | 4.6s（多任务共用卡时测得）/ 2.8s（独占时测得） | 5.4s |
| 端到端峰值显存（含文本编码器与 VAE）@512 | 16.0GB | 33.5GB |

数据来源：`baseline/<M>/eval_*`、`baseline/<M>/t2i_*/annotation_*`、`efficiency/*/summary.json`、`baseline/<M>/coco30k/timing.json`。

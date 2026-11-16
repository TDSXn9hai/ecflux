# 阶段3 块剪枝恢复实验：蒸馏能把删块的损伤恢复多少（2026-11-17）

- 学生：`outputs/stage2_v2/{full,no_block,fixed_ratio}/pruned`（stage2 v2 的剪枝 + 2000 步恢复结果）。`full` 与 `no_block` 的细粒度剪枝相同，差别只是 `full` 删了 Single 6/8/10/13 四个块。
- 教师：FLUX.2 Klein 9B Base，在线前向。损失沿用 `configs/distill.yaml`（vel + hid + attn，progressive），`ramp_start/ramp_end` 改为 500/2500。
- 预算：4000 步 × global batch 16（共 6.4 万个样本），lr 1e-5，cosine 衰减到 0。每 1000 步存一个 checkpoint。三个模型在 GPU1-3 上并行，每步约 8.1s，单个模型约 9h，显存峰值 58–60GB。
- 评估：与 APRA / drift_ref 的设置相同，用 D_val 前 200 条 prompt，512px、20 步、CFG 4.0、seed 42。脚本是 `scripts/eval_stage3.py`。step 0 重测得到的值与 stage2 `history.json` 的 `aux_final` 完全一致，原始 4B 的值与 `drift_ref/orig_metrics.json` 一致，说明两边口径相同。
- 运行命令：`bash scripts/run_stage3.sh {train,eval,dval,plot}`。

## 1. 结果

HPSv2 下降（200 条，原始 4B = 0.2830；数值越小越好）：

| 模型 | 参数比例 | step 0 | 1000 | 2000 | 3000 | 4000 | 恢复量 |
|---|---|---|---|---|---|---|---|
| full（删 4 块） | 0.746 | 0.0430 | 0.0488 | 0.0448 | 0.0421 | **0.0420** | 0.0010 |
| no_block | 0.862 | 0.0284 | 0.0235 | 0.0207 | 0.0203 | **0.0197** | 0.0087 |
| fixed_ratio（删 4 块） | 0.833 | 0.0371 | 0.0337 | 0.0324 | 0.0296 | **0.0295** | 0.0076 |
| full 与 no_block 的差距 | | 0.0146 | 0.0253 | 0.0241 | 0.0218 | **0.0223** | |

轨迹速度场 MSE（相对原始 4B，只作参考）：full 33.5 → 31.0，no_block 25.7 → 23.5，fixed_ratio 26.5 → 24.3。

D_val 1000 条（蒸馏前 → 蒸馏后；原始 4B 的 HPSv2 为 0.2825，CLIP 为 27.73）：

| 模型 | HPSv2 | CLIP |
|---|---|---|
| full | 0.2369 → 0.2399 | 26.97 → 27.39 |
| no_block | 0.2528 → 0.2605 | 27.31 → 27.68 |
| fixed_ratio | 0.2461 → 0.2513 | 27.14 → 27.68 |

9B 教师的 HPS200 是 0.2771，比原始 4B 低 0.006，图中用绿色虚线标出。按 HPS 计，9B 教师不是恢复上限。

## 2. 观察
1. **蒸馏几乎没有恢复 `full`。** `full` 的 HPS 下降只从 0.0430 降到 0.0420，`no_block` 降了 0.0087，同样删 4 块的 `fixed_ratio` 也降了 0.0076。`full` 和 `no_block` 的差距从 0.0146 扩大到 0.0223，D_val 1000 上的结论相同（差距 0.0159 → 0.0206）。
2. **`full` 的损伤不只来自删块数量。** `fixed_ratio` 删的块数和 `full` 一样，蒸馏后仍能恢复。`full` 在删块前的细粒度阶段已经剪到 0.862，再删块后就几乎恢复不动了。按目前的数据，问题在于细粒度剪枝加删块叠加后的容量，单独看块剪枝看不出这一点。
3. **曲线在 3000 步后变平，但不能据此断定已经饱和。** lr 按 cosine 在 4000 步衰减到 0，后 1000 步的有效学习率很低。`full` 在 1000 步时先变差，这和 500 步开始加入 hid/attn 损失的时间吻合。
4. **样例图（`samples/compare_step4000.jpg`）的目测：**
   - 三个剪枝模型都偏高饱和、高对比度，人脸带妆感。第 1 行的白色面部彩绘被画成整脸涂白，这在 `no_block` 上也有，属于剪枝本身的问题，与删块无关。
   - `full` 的结构扭曲更明显：鹿身体和角的结构混乱（第 4 行），皮卡车头变形（第 8 行），VR 场景里人物肢体错位（第 12 行），人脸也比另外两个模型更僵硬。
   - `fixed_ratio` 的轮廓介于两者之间。`no_block` 的结构最接近原始 4B。

## 3. 需要你判断的事
- 请看 `samples/compare_step{0,2000,4000}.jpg`，确认 `full` 蒸馏后的画质是否可以接受。按 stage2 的人工标定，HPS 下降 0.042 已经超过 ×2.0 档（2.5 级）。
- 如果不可接受，可以考虑这几个方向：
  - 不做块剪枝，以 `no_block` 为主线（参数 3.34B，离 2.5B 目标更远）
  - 改成 NanoFLUX 的连续块合并（Single 5–9 合成 1 块），在相同蒸馏预算下与本实验对比
  - 降低细粒度剪枝率，让出删块的容量（`fixed_ratio` 可以作为参考）
- 本实验的预算（6.4 万个样本）远小于开题设定的 20000 步 × 64，所以只能说明"短蒸馏恢复不了删块的损伤"，不能证明长时间蒸馏也不行。

## 4. 产物
- `<name>/step_{1000,2000,3000}`、`<name>/final`：蒸馏 checkpoint（含 `projectors.pt`）
- `<name>/metrics.json`：逐步数的 HPS200 和轨迹速度场 MSE
- `<name>/samples/step_<n>/`：12 条样例 prompt 的生成图，`<name>/eval_dval/`：final 在 D_val 1000 上的结果
- `_base4b/`、`_teacher9b/`：原始 4B 和 9B 教师的 HPS200 与样例图
- `fig_recovery.png`、`samples/compare_step{0,2000,4000}.jpg`、`summary.json`

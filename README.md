# ecflux：面向端云大小模型协同的 FLUX.2 Klein 端侧模型

开题报告《面向端云大小模型协同的端侧模型关键技术研究》的代码实现。以 FLUX.2 Klein 4B Base（5 Double + 20 Single Block，hidden 3072，24 头 × 128）为基线，三个研究内容对应三个阶段：

1. 多粒度结构化压缩（报告 4.2）：注意力头 / FFN 神经元 / Single Block 三粒度剪枝，加上自适应剪枝率调节 APRA → `scripts/prune.py`
2. 语义对齐知识蒸馏（报告 4.3）：速度场 + 隐藏层 + 跨模态注意力三级损失，教师为 FLUX.2 Klein 9B → `scripts/distill.py`
3. 任务复杂度判别器与端云协同（报告 4.4）：冻结 Text Encoder 嵌入 + 两层 MLP 路由，支持级联校验 → `scripts/train_router.py`、`scripts/collab_infer.py`

Text Encoder（Qwen3）与 VAE 全程冻结，只压缩和训练 DiT 主体。

## 目录结构

```
ecflux/
  flux_utils.py        模型加载、Qwen3 文本编码、Rectified Flow 加噪、Euler 采样、VAE 编解码
  hooks.py             FeatureRecorder：抓取隐藏状态、图像→文本注意力图、逐头文本注意力质量
  data.py              原始图文对读取、缓存数据集（支持多源按比例混合）
  generation.py        T2IGenerator：与 Flux2KleinPipeline 数值一致的推理封装
  metrics.py           CLIP Score / HPSv2 / FID / 参数量·峰值显存·延迟
  pruning/
    importance.py      头梯度敏感度、神经元激活 L1、块相似度、头模态类型
    pruner.py          表1 层级组、模态感知剪枝上界、块跳过选择
    prune_ops.py       物理删除头/神经元/块；剪枝模型保存与加载（prune_spec.json）
    evaluators.py      APRA 反馈信号：FID / velocity_mse / flow_loss
    apra.py            APRA 迭代 + 恢复训练 + 块剪枝阶段
  distill/
    losses.py          L_vel(robust) / L_hid / L_attn，递进式权重
    trainer.py         在线 / 离线教师蒸馏训练器、教师输出缓存
  router/
    features.py        提示词结构特征（对象数、属性、空间关系、文本渲染等）
    model.py           ComplexityRouter
    labels.py          师生质量差距 → 复杂度标签
    train.py           BCE 训练、预算约束阈值校准、成本质量曲线
    collab.py          端云协同推理（路由 / 级联 / 基线）
configs/               base / prune / distill / router 四份 YAML
scripts/               各阶段入口脚本 + run_ablations.sh + download_weights.sh
tests/                 smoke_test.py（库级）、smoke_scripts.py（脚本级）、real_weights_check.py
```

## 1. 环境

```bash
conda create -n ecflux python=3.11 -y && conda activate ecflux
# 先装与 CUDA 匹配的 torch/torchvision（示例为 CUDA 13.0）
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

硬件按报告设定为单卡 RTX 4090 24GB，显存优化靠 Gradient Checkpointing、AdamW 8-bit 和 bf16/fp16 混合精度。代码要求 Python ≥ 3.10、diffusers ≥ 0.37。报告里写的 diffusers 0.30+ 还没有 FLUX.2 Klein 的类，不能用。

## 2. 权重

```bash
bash scripts/download_weights.sh student   # 4B 管线 ≈ 16GB（已下载）
bash scripts/download_weights.sh eval      # CLIP / HPSv2.1 / OpenCLIP ViT-H / FID Inception ≈ 7.4GB（已下载）
bash scripts/download_weights.sh teacher   # 9B 管线 ≈ 35GB，gated，需先授权（见下）
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home   # 评估脚本运行前设置，评估模型缓存在这里
```

- 评估模型的缓存放在 `weights/hf_home` 和 `weights/torch_home`。系统盘 `/` 只剩约 3GB，不要用默认的 `~/.cache`。
- hpsv2 1.2.0 的 wheel 缺少 `hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz`，需要从 https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz 下载后放进该目录，否则 import 会报 FileNotFoundError。
- 国内网络可以设置 `export HF_ENDPOINT=https://hf-mirror.com`，或者配置 `https_proxy`。gated 仓库在镜像上同样需要 token。

### 9B 教师的下载步骤

1. 用 HF 账号登录，打开 https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B ，阅读并同意 FLUX 非商业许可，提交访问申请。通过后页面会显示可以访问。
2. 在 https://huggingface.co/settings/tokens 创建一个 Read 权限的 token。如果是 fine-grained token，要勾选 “Read access to contents of all public gated repos you can access”。
3. 在本机执行 `hf auth login`（或 `export HF_TOKEN=hf_xxx`），然后运行 `bash scripts/download_weights.sh teacher`。
4. 验证：`python -c "from huggingface_hub import hf_hub_download as d; print(d('black-forest-labs/FLUX.2-klein-base-9B','transformer/config.json'))"` 能正常打印路径就说明权限已生效。

教师选 `FLUX.2-klein-base-9B`（未蒸馏、需要 CFG），不选 `FLUX.2-klein-9B`（步数蒸馏版）。base 版和学生 4B Base 的训练目标、采样方式一致，速度场监督才有意义。

## 3. 数据准备

默认 `data.mode=online`：`prune.py` / `distill.py`（`teacher.mode=online`）直接读取 `data.sources` 中的原始图文（目录或 jsonl），训练时在线做 VAE 与文本编码，不需要下面的缓存步骤，也不占缓存磁盘。代价是要额外加载 VAE + Qwen3-4B（9B 教师还要 Qwen3-8B）文本编码器，每个 batch 多一次编码前向；显存不够时用 `--set data.encoder_device=cuda:1` 把编码器放到另一张卡。
```bash
python scripts/prune.py --config configs/prune.yaml --out outputs/prune/full --set "data.sources=[data/pexels_aes/index.jsonl]"
```
`teacher.mode=offline`（`cache_teacher.py`）本身就是缓存方案，仍需 `data.mode=cache`。

以下为 `data.mode=cache` 的流程。报告设定：从 LAION-Aesthetics V2 6.5+ 中选约 15 万对图文，提前缓存 VAE latent 和文本嵌入。

```bash
# (1) 用 img2dataset 下载为 「xxx.jpg + xxx.txt (+ xxx.json)」 目录格式，或者准备 jsonl：{"image":..., "caption":..., "aesthetic":...}
# (2) 缓存学生侧 latents + Qwen3-4B 嵌入
python scripts/cache_dataset.py --source data/laion_aes65 --out data/cache/laion_aes65 \
    --pipeline weights/FLUX.2-klein-base-4B --resolution 512 --min_aesthetic 6.5 --max_samples 150000
# (3) 9B 教师使用 Qwen3-8B，嵌入维度不同，需要额外缓存教师文本嵌入
python scripts/cache_dataset.py --stage teacher_text --out data/cache/laion_aes65 --pipeline weights/FLUX.2-klein-base-9B
# (4) 约 20% 的教师合成样本：先用教师生成图片，再按上面两步缓存为第二个目录
python scripts/generate.py --pipeline weights/FLUX.2-klein-base-9B --prompts data/synth_prompts.txt --out data/teacher_synth
#     generate.py 输出 index.jsonl，可整理成 {"image","caption"} jsonl 后交给 cache_dataset.py
#     训练时：--set "data.cache_dirs=[data/cache/laion_aes65,data/cache/teacher_synth]" "data.mix_ratios=[0.8,0.2]"
```

磁盘估算（单样本，bf16）：512px latent 为 128×32×32×2B ≈ 256KB；文本嵌入为 L×7680×2B，L=512 时约 7.5MB。15 万样本全长缓存需要 1TB 以上。建议用 `--max_sequence_length 128~256` 缩短，同时把 `data.max_sequence_length` 设成同一个值。

APRA 的 FID 反馈需要验证集：`data/val_prompts.txt` 放 1000 条 prompt（每行一条），`data/val_reference_images/` 放对应的参考图。

## 4. 阶段一：多粒度结构化压缩 + APRA

```bash
python scripts/prune.py --config configs/prune.yaml --out outputs/prune/full
# 产物：outputs/prune/full/pruned/{prune_spec.json, model.safetensors}  history.json（每轮 γ、FID、参数量、效率）
```

流程（`ecflux/pruning/apra.py::compress`）：

- 在校准 batch 上统计重要性。注意力头用梯度敏感度 $\sum_{W\in\{Q,K,V\}}\lVert \partial L/\partial W_i \odot W_i\rVert_1$，Double Block 同时累加文本流的 `add_q/k/v_proj`。FFN 神经元用 SwiGLU 激活的 L1 均值。相邻 Single Block 计算输出余弦相似度。
- 按表1 的六个层级组和当前剪枝率 γ_g 物理删除头和神经元。跨模态头最多剪 `alpha_cross=0.15`，自注意力头最多剪 `alpha_self=0.40`，FFN 最多剪 `beta_ffn=0.35`。
- 每轮剪枝后做 `recover.steps=2000` 步 Rectified Flow 恢复训练（lr=1e-5），然后在验证集上做 20 步采样并计算 FID。
- 按 ΔFID 更新 γ：小于 0.5 时乘 1.15，大于 2.0 时乘 0.80，否则乘 1.02，结果夹在 [γ_min, γ_max] 之间。
- 细粒度剪枝结束后，单独执行块级剪枝（相似度 > τ=0.92 的后继块删除），再恢复训练一次。

主要配置项（`configs/prune.yaml`）：

- `eval.metric`：`fid` 是报告设定。`velocity_mse`（相对原模型的速度场 MSE）和 `flow_loss` 是不需要采样的快速代理指标，适合调参。用代理指标时要同时调整 `apra.low/high`。
- `apra.target_param_ratio`：DiT 参数量降到原始的这个比例后提前停止。
- `prune.cross_ratio_threshold`：判定跨模态头的阈值，详见第 9 节。

只做剪枝、不蒸馏的模型 `outputs/prune/full/pruned` 就是表3 中的 Ours-PruneOnly 基线。

## 5. 阶段二：语义对齐知识蒸馏

```bash
# 在线教师（教师与学生同卡；24GB 放不下 9B bf16 教师加 4B 学生训练时，改用离线模式）
python scripts/distill.py --config configs/distill.yaml --out outputs/distill/full \
    --set student=outputs/prune/full/pruned teacher.mode=online

# 离线教师（报告 4.3.1 方案）：先缓存教师 v_T、隐藏状态、注意力图，训练时不加载教师
python scripts/cache_teacher.py --config configs/distill.yaml --out data/teacher_cache --set student=outputs/prune/full/pruned
python scripts/distill.py --config configs/distill.yaml --out outputs/distill/full \
    --set student=outputs/prune/full/pruned teacher.mode=offline teacher.offline_dir=data/teacher_cache
```

损失函数：$L=\lambda_{vel}L_{vel}+\lambda_{hid}L_{hid}+\lambda_{attn}L_{attn}$。

- $L_{vel}$：速度场 MSE。`robust=true` 时，学生输入加 $\delta\sim\mathcal N(0,0.01^2 I)$。
- $L_{hid}$：学生层 `double:3 / double:4 / single:11`（G2 末尾、G3 末尾、G5 中间）经过线性投影后，与教师对应层计算余弦距离。教师层位默认按相对深度自动映射（`teacher_hidden_layers: auto`）。
- $L_{attn}$：前 `attn_k` 个 Double Block 的图像 query → 文本 key 注意力图，计算 Frobenius² 距离。
- `progressive=true`：$\lambda_{hid}$ 和 $\lambda_{attn}$ 在 `ramp_start` 到 `ramp_end` 步之间从 0 线性升到目标值，实现由粗到细的递进训练。

训练设置与报告一致：AdamW 8-bit，lr 1e-5，余弦退火，梯度累积到全局 batch 64，开启 gradient checkpointing。产物 `final/` 和剪枝模型格式相同，另外附带 `projectors.pt`。

## 6. 阶段三：复杂度判别器与端云协同

```bash
P=data/router_prompts.txt   # GenEval / T2I-CompBench 的 prompt
# (1) 端侧学生和云端教师分别生成图片，并计算逐 prompt 的 CLIP / HPSv2
python scripts/generate.py --pipeline weights/FLUX.2-klein-base-4B --transformer outputs/distill/full/final --prompts $P --out outputs/gen/edge --steps 20
python scripts/generate.py --pipeline weights/FLUX.2-klein-base-9B --prompts $P --out outputs/gen/cloud --steps 20
python scripts/evaluate.py --images outputs/gen/edge  --metrics clip hps --out outputs/eval/edge
python scripts/evaluate.py --images outputs/gen/cloud --metrics clip hps --out outputs/eval/cloud
# (2) 根据质量差距构造标签（可选 --geneval_edge/--geneval_cloud 传入 GenEval 官方 results.jsonl）
python scripts/build_router_labels.py --config configs/router.yaml \
    --edge outputs/eval/edge/per_prompt.jsonl --cloud outputs/eval/cloud/per_prompt.jsonl --out data/router
# (3) 训练判别器，并按云端预算校准阈值 τ
python scripts/train_router.py --config configs/router.yaml --labels data/router/labels.jsonl --out outputs/router/default
# (4) 端云协同推理（mode: router / edge / cloud / random）
python scripts/collab_infer.py --config configs/router.yaml --prompts data/test_prompts.txt \
    --edge_transformer outputs/distill/full/final --router outputs/router/default/router.pt --out outputs/collab/router
python scripts/evaluate.py --images outputs/collab/router/images --metrics clip hps --out outputs/eval/collab_router
```

- 判别器复用端侧的 Qwen3 嵌入（masked mean pool），可以拼接 12 维结构特征，后接两层 MLP，用 BCE 训练。
- `calibration.cloud_budget` 是云端调用率上限，阈值取满足预算的最小 τ。`tradeoff_curve.csv` 给出不同 τ 下的云端调用率、召回率和系统平均 HPS。
- `collab.cascade=true` 时启用级联：$|p-\tau|<$ `band` 的请求先在端侧生成，HPSv2 低于 `hps_gate` 才转给云端重生成。
- `summary.json` 记录云端调用率、平均和 P95 时延、各路由的数量。

`collab_infer.py` 在本地加载教师来模拟云端。真实部署时，把 `cloud_fn` 换成远端服务客户端即可，`EdgeCloudSystem` 只要求它是 `(prompt, seed, emb) -> PIL.Image`。

## 7. 评估（表2）

```bash
python scripts/generate.py --pipeline weights/FLUX.2-klein-base-4B --transformer <DiT目录> --prompts coco30k.txt --out outputs/gen/X
python scripts/evaluate.py --images outputs/gen/X --ref_dir data/coco30k_images --metrics fid clip --out outputs/eval/X
python scripts/evaluate.py --images outputs/gen/X_hps --metrics hps --out outputs/eval/X   # HPSv2 benchmark prompts
python scripts/evaluate.py --transformer <DiT目录> --metrics efficiency --resolution 1024 --steps 20 --out outputs/eval/X
# GenEval（含 "text" 子集）/ T2I-CompBench：先按官方格式导出图片，再运行官方评测脚本
python scripts/generate.py ... --format geneval --prompts <geneval>/prompts/evaluation_metadata.jsonl --n_per_prompt 4 --out outputs/geneval/X
python scripts/generate.py ... --format t2icompbench --prompts <T2I-CompBench>/examples/dataset/spatial_val.txt --out outputs/t2i/X
```

表3 的对比对象：`weights/FLUX.2-klein-base-9B`（教师，即上限）、`weights/FLUX.2-klein-base-4B`（未压缩），以及 `outputs/prune/full/pruned`（Ours-PruneOnly）和 `outputs/distill/full/final`（Ours-Full）。所有对比都固定随机种子和采样步数（默认 20 步）。

## 8. 消融实验

`bash scripts/run_ablations.sh [prune|distill|router|all]` 会依次跑完下面所有实验。每个实验也可以单独通过 `--set` 覆盖配置来运行。

| 模块 | 消融项 | 配置覆盖 |
|---|---|---|
| 剪枝粒度 | 只剪头 / 只剪神经元 / 不剪块 | `prune.prune_neurons=false prune.prune_blocks=false` 等 |
| 头重要性 | 梯度敏感度 vs 权重幅值 vs 随机 | `prune.head_metric=grad\|magnitude\|random` |
| 神经元重要性 | 激活 L1 vs 权重幅值 vs 随机 | `prune.neuron_metric=activation\|magnitude\|random` |
| 模态感知 | 去掉跨模态头保护上界 | `prune.modality_aware=false` |
| APRA | 固定比例一次剪枝 | `apra.enabled=false` |
| APRA | 统一剪枝率（不分组） | `apra.uniform_rate=0.2` |
| APRA | 逐组敏感度反馈 / 失败回滚 | `apra.feedback=per_group`、`apra.rollback=true` |
| 块剪枝 | 解耦（之后）vs 之前 vs 不剪；阈值 τ | `block.stage=after\|before\|none`、`block.tau=0.90` |
| 蒸馏损失 | 逐项移除 L_vel / L_hid / L_attn | `distill.use_vel=false`（配合 `lambda_gt=1`）等 |
| 速度场 | 去掉噪声扰动 | `distill.robust=false` |
| 递进训练 | 三项损失从第 0 步同时生效 | `distill.progressive=false` |
| 注意力对齐 | 对齐的 Double Block 数 k | `distill.attn_k=1\|2\|3\|5` |
| 对齐形式 | 隐藏层 MSE、注意力 KL | `distill.hidden_kind=mse`、`distill.attn_kind=kl` |
| 判别器特征 | 仅嵌入 / 仅结构特征 / 两者拼接 | `router.features=emb\|struct\|emb+struct` |
| 云端预算 | 调用率上限 | `calibration.cloud_budget=0.1\|0.2\|0.3\|0.5` |
| 协同策略 | 全端侧 / 全云端 / 随机 / 路由 / 路由+级联 | `collab.mode=...`、`collab.cascade=false` |

## 9. 实现与报告的差异

下面几点是按 FLUX.2 Klein 的真实结构做的落地，或者是对报告中未定细节的取舍：

- FFN 激活：FLUX.2 实际使用 SwiGLU（`linear_in` 输出 [gate | up]），不是报告写的 GEGLU。神经元重要性按 SwiGLU 的输出 $a_j$ 统计，剪枝时 gate 行、up 行和 `linear_out` 列同步删除。
- 跨模态头的判定：MMDiT 是联合注意力，没有独立的 Cross-Attention 模块。代码统计每个头的图像 query 落在文本 key 上的注意力质量，相对均匀分配的倍数超过 `cross_ratio_threshold` 就判为跨模态头，受 α_cross 约束；其余头受 α_self 约束。对比实验用 `prune.modality_aware=false` 关闭这一判定。
- Single Block 结构：注意力与 MLP 并行，QKV 和 MLP-in 融合在同一个 `to_qkv_mlp_proj` 里。剪头和剪神经元都在这个融合矩阵的对应行上进行，`to_out` 同步删除对应列。
- 剪枝率的含义：γ_g 是相对原始模型的累计剪枝比例。每轮把各层剪到 ⌊γ_g·N_orig⌋，已经删掉的单元不会恢复。γ 下调时如果需要撤销本轮剪枝，打开 `apra.rollback`。
- APRA 的反馈粒度：按报告原式，一个全局 ΔFID 驱动所有组同步缩放，所以各组之间的比例不会变。为了让冗余度不同的组真正拉开差距，另外提供了 `feedback=per_group`：逐组临时剪枝探测敏感度，按组更新。
- 恢复训练的数据：D_val 只有 prompt，没有图像，所以恢复训练使用缓存训练集。D_val 只用于计算 FID。
- 加噪约定：采用 diffusers 的 $x_t=(1-t)x_0+t\epsilon$，$v=\epsilon-x_0$。报告里的 $x_t=x_0+t\cdot v$ 是这条直线插值的简写。
- 注意力图对齐：9B 教师和剪枝后学生的头数不同，默认先对头取平均再比较（`attn_head_mode=mean`）。教师是同构的原始 4B 时，可以用 `match` 按原始头编号逐头对齐。图像 query 默认按步长 2 下采样，以节省显存。Frobenius² 默认按元素归一化（`attn_reduction=mean`），这样 λ 与分辨率无关。
- 师生文本编码器不同：Klein 9B 使用 Qwen3-8B（joint_attention_dim 不同），所以缓存里另外保存 `teacher_prompt_embeds`。两者 tokenizer 相同，文本 token 数一致，注意力图可以直接对齐。
- GenEval、T2I-CompBench、HPSv2 benchmark 依赖各自官方仓库的检测器或评测流程，本仓库只负责按格式导出图片。

## 10. 冒烟测试

本机没有 GPU。以下三项测试都在 CPU 上运行过并全部通过，没有做真实训练：

```bash
python tests/smoke_test.py         # 库级：剪枝算子数值等价、APRA 五种配置、蒸馏 7 种配置（含离线缓存）、采样、路由与协同、端到端生成
python tests/smoke_scripts.py      # 脚本级：prune → distill(online) → cache_teacher → distill(offline) → evaluate → 标签 → 路由训练
python tests/real_weights_check.py weights/FLUX.2-klein-base-4B   # 真实 4B DiT（约 10GB 内存）：重要性 → 剪枝 → 前向 → spec 重建
```

库级和脚本级测试使用与 Klein 同拓扑（5 Double + 20 Single）的极小随机模型。真实权重检查的结果：3.876B 参数按表1 初始剪枝率剪枝并删除 2 个块后变为 3.021B，前向输出形状正确，由 spec 重建的结构与原模型一致。

另外在 CPU 上单独验证过：Qwen3-4B 文本编码（输出 `(B, L, 7680)`）、CLIP Score、HPSv2.1 打分、FID（离线加载 Inception）。

没有验证的部分（需要 GPU 或 9B 权重）：9B 教师、24GB 显存下的实际占用、训练收敛效果。

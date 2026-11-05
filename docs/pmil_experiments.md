# DFD-HR + Part-aware MIL Fusion (PMIL)：实验设计与运行命令

## 方法一句话

DFD-HR 的 local 分支输出 32×32 个 patch token，但原模型分类只用了 4 个 split 的 [CLS]。
PMIL 用一个固定的人脸部位模板（眼眉 / 鼻 / 嘴 / 轮廓融合带 / 脸颊）给 K=5 个 query 加注意力先验，
每个部位独立给出伪造 logit，再按多示例学习（真脸 = 所有部位都真；假脸 = 至少一个部位假）用 smooth-max 聚合，
最终 logit = 0.5·原 DFD-HR logit + 0.5·部位 bag logit。

## 评测协议（所有实验一致）

| 测试集 | 数据来源 | 方式 |
|---|---|---|
| CDF-v2 (`Celeb-DF-v2`)、FSh (`FaceShifter`)、DFD (`DeepFakeDetection`)、DFDC、DFDCP | `/data/hjr/dfbench/rgb` + JSON manifest | `scripts/test.py` |
| WDF | `/data/hjr/deepfake/testset/WDF` | `scripts/evaluate_deepfake_testset.py` |

- 每视频 32 帧，448 输入，主指标 Video AUC，同时记录 Frame AUC。
- 训练：FF++ c23，每视频 8 帧；按 Celeb-DF-v2 选 best checkpoint（与 `scripts/run_dfd_hr.sh` / 原论文代码一致）。
- 一键评测：`scripts/run_pmil_eval.sh`（4 卡并行，自动汇总 6 个数据集与平均值）。

## 实验列表

### Exp-0 基线（只评测，不训练）
官方 `checkpoints/ckpt_best.pth` 用**同一评测脚本**重新跑 6 个数据集，作为所有 Δ 的参照行。

### Exp-1 主实验（最快）：冻结 DFD-HR，只训练 PMIL 分支
- 配置：`training/config/detector/dfd_hr_pmil.yaml`
- 从官方 checkpoint 初始化并冻结全部原参数，只训练 PMIL（约 1M 参数），4 个 epoch。
- 原分支与官方权重逐位相同，所以 AUC 的变化只可能来自新分支；也不存在“多训了几个 epoch”的混淆。
- 冻结部分前向在 `no_grad` + eval 模式下运行（层早退是确定性的，与推理一致），训练很快。

### 主要消融（都在 Exp-1 设置下，便宜）
| 编号 | 配置 | 验证什么 |
|---|---|---|
| A1 | `pmil/abl_free_queries.yaml` | 去掉部位先验（QTFP 式自由 query）→ 部位先验是否必要 |
| A2 | `pmil/abl_mean_agg.yaml` | MIL smooth-max → 平均 → “任一部位假即假”的聚合是否必要 |
| A3 | `pmil/abl_no_mil_loss.yaml` | 去掉“真脸每个部位都真”的约束 → MIL 损失是否必要 |
| A4 | `pmil/abl_no_boundary.yaml` | 去掉轮廓融合带部位 → 融合边界线索的贡献 |
| A5 | `pmil/abl_single_query.yaml` | 只用 1 个整脸 query → 部位分解本身是否必要（排除“只是多了一个 patch-token 分支”） |
| E1 | `pmil/eval_part_only.yaml`（只评测） | 只用部位分支的 AUC |
| E2 | `pmil/eval_orig_only.yaml`（只评测） | 只用原分支：必须等于 Exp-0，作为正确性检查 |

### Exp-2（论文级，慢）：端到端从头训练
- 配置：`pmil/e2e_scratch.yaml`（20 epoch，与 DFD-HR 完全相同的训练协议）
- 必须同时用同一脚本、同一 seed 重训 DFD-HR 基线（`training/config/detector/dfd_hr.yaml`），用两者对比。
- 建议 Exp-1 有明显提升后再跑。

## 运行命令

```bash
cd /path/to/dfd-hr && git fetch origin && git checkout pmil

# 0) 生成部位模板 + 检查各数据集裁剪是否对齐（看 outputs/part_template/check_*.png）
python3 scripts/build_part_template.py \
  --check "FF=/data/hjr/dfbench/rgb/FaceForensics++/original_sequences/youtube/c23/frames/*/*.png" \
  --check "CDF=/data/hjr/dfbench/rgb/Celeb-DF-v2/Celeb-synthesis/frames/*/*.png" \
  --check "WDF=/data/hjr/deepfake/testset/WDF/*/*/*/*/*.png"

# 1) Exp-0 基线（同协议）
CHECKPOINT=checkpoints/ckpt_best.pth DETECTOR=training/config/detector/dfd_hr.yaml \
RUN_ID=baseline_6set bash scripts/run_pmil_eval.sh

# 2) Exp-1 主实验（训练结束后自动跑 6 个测试集）
RUN_ID=pmil_main bash scripts/run_pmil_train.sh

# 3) 对比
python3 scripts/summarize_video_auc.py \
  outputs/pmil/eval/baseline_6set/pmil_baseline_6set.json \
  outputs/pmil/eval/pmil_main_eval/pmil_pmil_main_eval.json
```

消融（主实验有提升后再跑）：

```bash
for a in free_queries mean_agg no_mil_loss no_boundary single_query; do
  DETECTOR=training/config/detector/pmil/abl_${a}.yaml RUN_ID=abl_${a} bash scripts/run_pmil_train.sh
done

# 只评测的两个变体（用主实验 checkpoint）
CKPT=outputs/pmil/train/pmil_main/test/avg/ckpt_best.pth
CHECKPOINT=$CKPT DETECTOR=training/config/detector/pmil/eval_part_only.yaml RUN_ID=E1_part_only bash scripts/run_pmil_eval.sh
CHECKPOINT=$CKPT DETECTOR=training/config/detector/pmil/eval_orig_only.yaml RUN_ID=E2_orig_only bash scripts/run_pmil_eval.sh

python3 scripts/summarize_video_auc.py outputs/pmil/eval/baseline_6set/*.json outputs/pmil/eval/*_eval/*.json \
  outputs/pmil/eval/E1_part_only/*.json outputs/pmil/eval/E2_orig_only/*.json
```

## 结果不理想时的调整顺序

1. `check_*.png` 中部位轮廓明显偏离某个数据集的平均脸 → 模板与该数据集裁剪不一致，先解决对齐。
2. `E1` 部位分支单独 AUC 很低 → 加大 `nEpochs`，或把 `mask_eps` 从 0.02 调到 0.05（先验更软）。
3. `E1` 不低但融合后提升小 → 两个分支相关性太高，改跑 Exp-2 端到端。
4. 不要在测试集上调 `fuse_alpha`；它固定为 0.5。

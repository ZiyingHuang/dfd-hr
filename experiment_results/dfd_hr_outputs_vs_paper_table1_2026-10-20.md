# DFD-HR 实验结果与原论文 Table 1 对比

## 1. 报告范围

本报告整理并对比以下两次测试结果：

### Folder-label deepfake 测试

```text
/home/huangjunrong/deepfake/main/DFD-HR/outputs/deepfake/eval/20261019_115030/
```

- 数据根目录：`/data/hjr/deepfake/testset`
- 标签来源：目录名 `real/fake`
- Checkpoint：`checkpoints/ckpt_best.pth`
- 测试采样：每个视频最多 32 帧
- Batch size：32
- 数据集：CDF、DFD、DFDC、DFDCP、DFR、FFIW、WDF

### 帧使用口径说明

Folder-label evaluator 并不是无条件读取原始视频的全部帧，而是使用：

```text
每个序列最多 32 帧
```

其逻辑是：如果序列帧数不超过 32，则使用该序列目录中的全部已提取帧；如果超过 32，则均匀采样最多 32 帧。

本次 `/data/hjr/deepfake/testset` 的实际帧数范围如下：

| 数据集 | 每视频帧数范围 | 是否超过 32 帧 |
|---|---:|---:|
| CDF | 20 | 否 |
| DFD | 5 | 否 |
| DFDC | 5–6 | 否 |
| DFDCP | 20 | 否 |
| DFR | 20 | 否 |
| FFIW | 10 | 否 |
| WDF | 20–22 | 否 |

因此，本次 folder-label 结果使用了各目录中已有的全部已提取帧，但这些帧不一定等于原始视频的全部原始帧。

dfbench 测试则从 JSON manifest 的候选帧列表中最多采样 32 帧。当前 manifest 每个视频通常约有 32 个候选帧，因此多数情况下使用全部 manifest 帧，但同样不等于读取原始视频的全部帧。

### JSON-based DeepfakeBench/DFD-HR 测试

```text
/home/huangjunrong/deepfake/main/DFD-HR/outputs/dfbench/eval/20261018_114206/
```

- RGB 根目录：`/data/hjr/dfbench/rgb`
- JSON manifest：`/data/hjr/dfbench/dataset_json`
- Checkpoint：`checkpoints/ckpt_best.pth`
- `frame_num_test`：32
- 数据集：Celeb-DF-v2、DeepFakeDetection、DFDC、DFDCP

## 2. 原论文 Table 1 基线

原论文 Table 1 的 Protocol-1 使用 FF++ c23 训练，并报告 Video-level AUC：

| 论文数据集 | 论文 Video AUC |
|---|---:|
| CDF-v2 | 0.960 |
| DFD | 0.980 |
| DFDC | 0.865 |
| DFDCP | 0.901 |
| DFo / DeeperForensics | 0.997 |
| WDF | 0.907 |
| FFIW | 0.968 |
| **论文 Avg.** | **0.940** |

说明：论文 Table 1 不包含 FaceShifter；FaceShifter 不是本报告的 Table 1 对比项。

来源：

- [DFD-HR CVPR 2026 原论文 Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_DFD-HR_Generalizable_Deepfake_Detection_via_Hierarchical_Routing_Learning_CVPR_2026_paper.pdf#page=5)
- [DFD-HR 官方 Table 1 图片](https://github.com/Disguiser15/DFD-HR/blob/ec85fc462d06f06448d2222748e24e7113f25784/misc/deepfake_tab1.png)

论文实验协议为训练每视频 8 帧、推理每视频 32 帧、输入分辨率 448。原始论文的主要比较指标是 Video AUC。

## 3. Folder-label deepfake 结果

指标顺序与 JSON 一致：Frame AUC、Frame ACC、Frame EER、Frame AP、Video AUC、Video ACC、Video EER。

| 数据集 | 视频数 | 帧数 | Frame AUC | Frame ACC | Frame EER | Frame AP | Video AUC | Video ACC | Video EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDF | 518 | 10,360 | 0.874447 | 0.794595 | 0.208146 | 0.929117 | **0.940664** | 0.853282 | 0.157303 |
| DFD | 3,427 | 17,135 | 0.947766 | 0.905807 | 0.128375 | 0.993416 | **0.971951** | 0.936971 | 0.085399 |
| DFDC | 4,942 | 24,711 | 0.835141 | 0.745700 | 0.251075 | 0.851986 | **0.874029** | 0.784096 | 0.211679 |
| DFDCP | 748 | 14,960 | 0.850955 | 0.770053 | 0.236553 | 0.921466 | **0.898377** | 0.828877 | 0.178030 |
| DFR | 402 | 8,040 | 0.986187 | 0.940174 | 0.056468 | 0.986371 | **0.997970** | 0.980100 | 0.009950 |
| FFIW | 1,831 | 18,310 | 0.894277 | 0.836210 | 0.192085 | 0.496821 | **0.948128** | 0.883124 | 0.134027 |
| WDF | 806 | 16,122 | 0.862134 | 0.738618 | 0.217677 | 0.875654 | **0.899076** | 0.751861 | 0.186869 |

### 与论文 Table 1 的 Video AUC 对比

| 本地名称 | 论文对应项 | 本地 Video AUC | 论文 Video AUC | 差值（本地−论文） |
|---|---|---:|---:|---:|
| CDF | CDF-v2 | 0.940664 | 0.960 | -0.019336 |
| DFD | DFD / DeepFakeDetection | 0.971951 | 0.980 | -0.008049 |
| DFDC | DFDC | 0.874029 | 0.865 | +0.009029 |
| DFDCP | DFDCP | 0.898377 | 0.901 | -0.002623 |
| DFR | DFo / DeeperForensics | 0.997970 | 0.997 | +0.000970 |
| FFIW | FFIW | 0.948128 | 0.968 | -0.019872 |
| WDF | WDF | 0.899076 | 0.907 | -0.007924 |
| **7 项本地算术平均** | **论文 Avg.** | **0.932885** | **0.940** | **-0.007115** |

其中 CDF、DFD、DFR 是根据本地 folder 名称做的对应映射，名称并非 JSON 中的完整论文名称，应谨慎解读。

## 4. JSON-based dfbench 结果

| 数据集 | Frame AUC | Frame ACC | Frame EER | Frame AP | Video AUC | Video ACC | Video EER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | 0.884502 | 0.810536 | 0.200356 | 0.934554 | **0.948860** | 0.853282 | 0.140449 |
| DeepFakeDetection | 0.942964 | 0.909041 | 0.119449 | 0.993121 | **0.973720** | 0.942840 | 0.088154 |
| DFDC | 0.858504 | 0.747328 | 0.230301 | 0.880696 | **0.892725** | 0.763818 | 0.192225 |
| DFDCP | 0.883546 | 0.800081 | 0.201661 | 0.940683 | **0.918040** | 0.835890 | 0.160870 |
| Celeb-DF-v1 | 0.885383 | 0.773916 | 0.189526 | 0.924705 | **0.945671** | 0.760000 | 0.105263 |
| FaceShifter | 0.862506 | 0.785220 | 0.217906 | 0.874181 | **0.925969** | 0.835714 | 0.164286 |
| UADFV | 0.979278 | 0.786705 | 0.046512 | 0.986208 | **0.984173** | 0.795918 | 0.020408 |
| **7 项本地算术平均** | **0.899526** | **0.801832** | **0.172244** | **0.933450** | **0.941308** | **0.826780** | **0.124522** |

新增三项测试来自：

```text
outputs/dfbench/eval/dfbench_extra_20261026_081527/
```

- `CDF-v1` 对应 `Celeb-DF-v1`；
- `FSh` 对应 `FaceShifter`；
- 三项测试使用原 dfbench JSON 配置、32 帧上限和 `ckpt_best.pth`；
- FaceShifter、Celeb-DF-v1 和 UADFV 不属于论文 Table 1 的四项重叠子集，因此 7 项平均仅作为本地扩展统计。

该批 dfbench **结果 JSON** 没有记录 `num_frames`、`num_videos`、real/fake 视频数，因此不能仅从结果 JSON 恢复测试样本数量。下面的序列/帧统计是对独立的 `/data/hjr/dfbench/dataset_json/*.json` 测试 manifest 重新解析得到的；日志中的 batch 数量也不能直接等同于视频数或帧数。

### dfbench 测试 manifest 的每序列帧数

`/data/hjr/dfbench/dataset_json/*.json` 中，测试序列的 manifest 帧数不是固定 32 帧，而是最多 32 帧：

| 数据集 | 测试序列数 | 每序列帧数范围 | 恰好 32 帧的序列 | manifest 总帧数 |
|---|---:|---:|---:|---:|
| Celeb-DF-v2 | 518 | 3–32 | 476 | 16,420 |
| DeepFakeDetection | 3,429 | 1–32 | 1,584 | 102,541 |
| DFDC | 4,704 | 1–32 | 3,338 | 132,116 |
| DFDCP | 654 | 1–32 | 339 | 17,222 |
| **合计** | **9,305** | **1–32** | **5,737** | **268,299** |

由于这些 JSON manifest 中没有序列超过 32 帧，当前 DFD-HR 测试配置 `frame_num.test = 32` 会使用每个序列 manifest 中的全部候选帧。这里的帧是 manifest 已经预先记录的候选帧，不一定等于原始视频的全部原始帧。

### 与论文 Table 1 的 Video AUC 对比

| 数据集 | 本地 Video AUC | 论文 Video AUC | 差值（本地−论文） |
|---|---:|---:|---:|
| Celeb-DF-v2 | 0.948860 | 0.960 | -0.011140 |
| DeepFakeDetection | 0.973720 | 0.980 | -0.006280 |
| DFDC | 0.892725 | 0.865 | +0.027725 |
| DFDCP | 0.918040 | 0.901 | +0.017040 |
| **4 项本地算术平均** | **0.933336** | **0.926500*** | **+0.006836** |

`*` 0.926500 是四个重叠数据集的论文值算术平均，不是论文 Table 1 的七数据集 Avg. 0.940。

## 5. 结果解读

### Folder-label deepfake 结果

- 7 项平均 Video AUC 为 `0.932885`，低于论文 Table 1 发布的 `0.940`。
- CDF 和 FFIW 的差距较大，分别为 `-0.019336` 和 `-0.019872`。
- DFR/DeeperForensics 结果基本匹配论文，差值为 `+0.000970`。
- 该结果来自 `/data/hjr/deepfake/testset` 的文件夹扫描，不是 DFD-HR 官方 JSON manifest 测试流程。

### JSON-based dfbench 结果

- 原有四个重叠数据集的本地平均 Video AUC 为 `0.933336`。
- 加入 Celeb-DF-v1、FaceShifter、UADFV 后，7 项本地扩展平均 Video AUC 为 `0.941308`。
- 相对这四个数据集的论文子集平均值 `0.926500`，本地结果高 `0.006836`。
- DFDC 和 DFDCP 高于论文值，Celeb-DF-v2 和 DeepFakeDetection 低于论文值。
- 该批结果使用 DFD-HR JSON manifest/RGB 目录，测试元数据记录为 32 帧，更接近论文的推理帧数设置。

## 6. 为什么不能直接判定“超过/复现论文”

两组本地结果都应视为 checkpoint 的本地复测，而不是严格的论文复现，原因包括：

1. `deepfake` 组使用 folder-label 数据集，`dfbench` 组使用 JSON manifest；数据来源和样本划分不同。
2. 本地 folder-label 结果的视频数与论文报告值不完全一致：
   - CDF：518 vs 论文 518；
   - DFD：3,427 vs 论文 3,429；
   - DFDC：4,942 vs 论文 4,704；
   - DFDCP：748 vs 论文 654。
3. CDF、DFD、DFR 的名称是本地目录命名，需要映射到论文名称，不能假定完全相同的测试子集。
4. 两组 JSON 都没有 Video AP；论文 Table 1 的主要指标是 Video AUC，因此本报告只用 Video AUC 做主比较。
5. 原论文 Table 1 的结果来自论文实验环境和官方数据划分，本地 checkpoint、预处理、路径 manifest 和运行环境可能不同。

## 7. 结论

- **dfbench JSON-based 结果**：原有四个重叠数据集 Video AUC 为 `0.948860 / 0.973720 / 0.892725 / 0.918040`，四项平均 `0.933336`；新增三项后，7 项本地扩展平均为 `0.941308`。由于新增数据集不属于论文 Table 1 的同一比较集合，不能将 `0.941308` 直接视为论文结果复现值。
- **deepfake folder-label 结果**：七项平均 Video AUC 为 `0.932885`，相对论文 Table 1 的 `0.940` 低 `0.007115`。
- 若要进行严格论文复现，应统一使用论文指定的 JSON 划分、FF++ c23 训练设置、32 帧推理、相同预处理和相同视频级聚合方式，并补齐所有官方测试 manifest。

## 8. 原始结果文件

### deepfake

```text
outputs/deepfake/eval/20261019_115030/20261019_115030_summary.json
outputs/deepfake/eval/20261019_115030/20261019_115030_{CDF,DFD,DFDC,DFDCP,DFR,FFIW,WDF}.json
```

### dfbench

```text
outputs/dfbench/eval/20261018_114206/test_metrics_20261018_123441_Celeb-DF-v2.json
outputs/dfbench/eval/20261018_114206/test_metrics_20261018_123518_DFDCP.json
outputs/dfbench/eval/20261018_114206/test_metrics_20261018_170312_DeepFakeDetection.json
outputs/dfbench/eval/20261018_114206/test_metrics_20261018_183618_DFDC.json
outputs/dfbench/eval/dfbench_extra_20261026_081527/test_metrics_dfbench_extra_20261026_081527_Celeb-DF-v1.json
outputs/dfbench/eval/dfbench_extra_20261026_081527/test_metrics_dfbench_extra_20261026_081527_FaceShifter.json
outputs/dfbench/eval/dfbench_extra_20261026_081527/test_metrics_dfbench_extra_20261026_081527_UADFV.json
```

本报告只读这些结果文件生成，未修改任何数据集、JSON manifest 或实验结果 JSON。

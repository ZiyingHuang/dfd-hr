# DFD-HR DF40 Cross-method Evaluation 结果对比

## 1. 实验概况

本报告整理以下 DF40 测试结果，并与 DFD-HR 论文 Table 1 的 **Cross-method Evaluation (Protocol-2)** 对比：

```text
/home/huangjunrong/deepfake/main/DFD-HR/outputs/dfbench/eval/df40_20261026_091657/
```

- Checkpoint：`/home/huangjunrong/deepfake/main/DFD-HR/checkpoints/ckpt_best.pth`
- 数据根目录：`/data/hjr/df40/test`
- JSON manifest：`/data/hjr/df40/test/dataset_json`
- 测试配置：DF40 JSON 配置，`frame_num.test = 32`
- 指标：Frame-level AUC/ACC/EER/AP，以及 Video-level AUC/ACC/EER
- 论文 Table 1 的主比较指标：**Video-level AUC**

本次测试包含完整的 8 个 Cross-method 子集：UniFace、BlendFace、MobileSwap、e4s、FaceDancer、FSGAN、InSwapper 和 SimSwap。

## 2. 数据集名称对应关系

| 论文列名 | 本地 JSON 数据集名 | DF40 方法 |
|---|---|---|
| UniFace | `uniface_ff` | UniFace |
| BleFace | `blendface_ff` | BlendFace |
| MobSwap | `mobileswap_ff` | MobileSwap |
| e4s | `e4s_ff` | e4s |
| FaceDan | `facedancer_ff` | FaceDancer |
| FSGAN | `fsgan_ff` | FSGAN |
| InSwap | `inswap_ff` | InSwapper |
| SimSwap | `simswap_ff` | SimSwap |

`ff` 表示 DF40 的 FF++ 数据域。每个 JSON 同时包含该方法对应的 Real 和 Fake 条目；本次推理使用每个视频最多 32 个 manifest 帧。

## 3. 本地结果与论文结果

### 3.1 完整指标

| 数据集 | Frame AUC | Frame ACC | Frame EER | Frame AP | Video AUC | Video ACC | Video EER |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniFace | 0.965415 | 0.890877 | 0.098236 | 0.966340 | **0.987653** | 0.925000 | 0.042857 |
| BlendFace | 0.899820 | 0.813995 | 0.186872 | 0.901388 | **0.955663** | 0.875000 | 0.135714 |
| MobileSwap | 0.946421 | 0.887713 | 0.126367 | 0.990167 | **0.987188** | 0.950902 | 0.057143 |
| e4s | 0.977971 | 0.899902 | 0.066756 | 0.977812 | **0.990311** | 0.933333 | 0.042857 |
| FaceDancer | 0.943201 | 0.868188 | 0.127261 | 0.949090 | **0.980612** | 0.921429 | 0.064286 |
| FSGAN | 0.958014 | 0.882471 | 0.102032 | 0.966652 | **0.984847** | 0.928571 | 0.064286 |
| InSwapper | 0.956005 | 0.875986 | 0.114534 | 0.952498 | **0.986622** | 0.922481 | 0.078571 |
| SimSwap | 0.937043 | 0.861630 | 0.136414 | 0.937216 | **0.978010** | 0.917857 | 0.078571 |
| **8 项本地算术平均** | **0.947986** | **0.872595** | **0.119809** | **0.955145** | **0.981363** | **0.921822** | **0.070536** |

### 3.2 Video AUC 与论文 Table 1 对比

论文 Table 1 Cross-method Evaluation 的 DFD-HR（Ours）结果为：

```text
UniFace 0.988, BleFace 0.956, MobSwap 0.986, e4s 0.993,
FaceDan 0.970, FSGAN 0.984, InSwap 0.981, SimSwap 0.970,
Avg. 0.978
```

| 数据集 | 本地 Video AUC | 论文 Video AUC | 差值（本地−论文） |
|---|---:|---:|---:|
| UniFace | 0.987653 | 0.988 | -0.000347 |
| BlendFace | 0.955663 | 0.956 | -0.000337 |
| MobileSwap | 0.987188 | 0.986 | +0.001188 |
| e4s | 0.990311 | 0.993 | -0.002689 |
| FaceDancer | 0.980612 | 0.970 | +0.010612 |
| FSGAN | 0.984847 | 0.984 | +0.000847 |
| InSwapper | 0.986622 | 0.981 | +0.005622 |
| SimSwap | 0.978010 | 0.970 | +0.008010 |
| **8 项平均** | **0.981363** | **0.978500** | **+0.002863** |

## 4. 结果分析

- 本地 8 项平均 Video AUC 为 **0.981363**，论文 8 项平均值为 **0.978500**，本地结果高 **0.002863**。
- 本地 Video AUC 高于论文值的数据集有：MobileSwap、FaceDancer、FSGAN、InSwapper 和 SimSwap。
- 本地 Video AUC 略低于论文值的数据集有：UniFace、BlendFace 和 e4s。
- 最大正差值是 FaceDancer：`+0.010612`。
- 最大负差值是 e4s：`-0.002689`。
- 本地结果与论文结果整体接近，8 个方法中有 7 个数据集的绝对差值不超过 `0.01`；FaceDancer 的差值略超过 `0.01`。

## 5. 与论文结果的可比性说明

本次结果可以作为 DFD-HR Cross-method Evaluation 的本地复测，但不应直接宣称为完全复现，原因包括：

1. 论文结果来自作者发布的实验环境和数据处理流程；本地使用的是当前容器中的 DF40 处理数据和 JSON manifest。
2. 本地 Fake 数据部分来自 DF40 处理后镜像，Real 数据来自 `FaceForensics++_real_data_for_DF40.zip`，路径和文件组织已按 JSON 映射验证。
3. 本地使用的 checkpoint 是：
   ```text
   checkpoints/ckpt_best.pth
   ```
4. 论文表格报告的是 Video-level AUC；本报告同时保留 Frame-level 指标，但论文对比只使用 Video AUC。
5. 平均值是 8 个子集的简单算术平均，不是按视频数量加权的总体 AUC。

## 6. 原始结果文件

```text
outputs/dfbench/eval/df40_20261026_091657/
└── test_metrics_20261026_091657_uniface_ff_blendface_ff_mobileswap_ff_e4s_ff_facedancer_ff_fsgan_ff_inswap_ff_simswap_ff.json
```

论文来源：DFD-HR CVPR 2026 Table 1，Cross-method Evaluation (Protocol-2)。

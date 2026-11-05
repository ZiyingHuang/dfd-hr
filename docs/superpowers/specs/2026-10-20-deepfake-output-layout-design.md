# Deepfake 输出目录整理设计

## 目标

将 DFD-HR 的训练和评估结果按数据源统一整理到：

```text
outputs/deepfake/
├── train/<run_id>/
└── eval/<run_id>/

JSON-based DeepfakeBench 结果使用同样的结构：

```text
outputs/dfbench/
├── train/<run_id>/
└── eval/<run_id>/
```
```

其中 `<run_id>` 默认使用 `YYYYMMDD_HHMMSS`，也允许通过环境变量或命令行显式指定。

## 现状

- folder-based deepfake evaluator 当前把 JSON、summary 和 GPU 日志直接写入 `outputs/deepfake/`。
- JSON-based paper evaluator 当前写入 `outputs/dfbench/`。
- 训练日志和 checkpoint 默认写入 `logs/` 下的模型时间目录。
- `outputs/deepfake/` 根目录已有两次历史 eval：`20261019_031304` 和 `20261019_115030`。

## 设计

### 评估输出

- `scripts/evaluate_deepfake_testset.py` 默认输出到 `outputs/deepfake/eval/<run_id>/`。
- `scripts/run_deepfake_testset_4gpu.sh` 将 worker 日志、每数据集 JSON 和 summary 统一写入同一个 eval run 目录。
- `scripts/run_paper_eval_4gpu.sh` 将 JSON-based DFD-HR/DeepfakeBench 测试结果和 GPU 日志写入 `outputs/dfbench/eval/<run_id>/`。
- `scripts/test.py` 默认输出目录改为 `outputs/dfbench/eval/<run_id>/`，显式 `--output_dir` 时保持兼容。

### 训练输出

- `scripts/run_dfd_hr.sh` 默认生成 `outputs/dfbench/train/<run_id>/` 并传给训练入口；`OUTPUT_GROUP` 可切换输出命名空间。
- `scripts/train.py` 接收 `--output_dir`，将 `config['log_dir']` 设置为该目录。
- `Trainer` 保留现有 `train/`、`test/`、TensorBoard、checkpoint 和 metric 子目录，但其根目录改为传入的 run 目录。
- 训练 run 目录中的 `training.log`、checkpoint、特征和数据字典保持现有文件命名。

### 历史文件迁移

只移动 `outputs/deepfake/` 根目录中的历史 eval 文件，不修改任何数据集：

```text
outputs/deepfake/20261019_031304_* → outputs/deepfake/eval/20261019_031304/
outputs/deepfake/20261019_115030_* → outputs/deepfake/eval/20261019_115030/
```

原始文件名保持不变；不为失败的数据集补造 JSON。迁移后 `outputs/deepfake/` 根目录不再放置结果文件。

## 兼容性与安全性

- 显式 `--output_dir` 参数优先级高于默认目录。
- 显式 `RUN_ID` 可复现或关联外部任务；重复使用同一 run id 前应确认不会覆盖同名结果。
- 数据集目录保持只读，迁移只涉及 DFD-HR 的输出目录。
- folder-based deepfake 结果写入 `outputs/deepfake/`；JSON-based benchmark 结果写入 `outputs/dfbench/`。

## 验证

- Python 脚本通过编译检查。
- shell wrapper 通过 `bash -n`。
- 单元测试覆盖默认路径、显式输出目录和 run 目录创建。
- 迁移后核对每个历史 run 的文件数量和文件内容未改变。

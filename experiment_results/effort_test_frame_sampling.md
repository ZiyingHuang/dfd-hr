# Effort 测试集抽帧策略说明

## 1. 当前测试配置

Effort 当前配置为：

```yaml
frame_num: {'train': 8, 'test': 32}
```

论文协议是训练阶段每个视频使用 8 帧，推理阶段使用 32 帧。

## 2. 测试阶段实际行为

代码位置：

```text
DeepfakeBench/training/dataset/abstract_dataset.py
```

数据集 JSON 已经为每个视频保存了待读取的帧路径。测试阶段首先读取：

```python
frame_paths = video_info['frames']
```

之后根据 `frame_num['test']` 判断是否需要继续采样。

当前 `/data/hjr/dfbench/dataset_json` 中每个视频最多只有 32 个帧路径，因此实际行为是：

- 视频有 32 帧：直接使用全部 32 帧；
- 视频少于 32 帧：使用全部已有帧；
- 不会在测试阶段从原始视频重新读取并计算抽帧间隔；
- 不存在统一的“每隔 N 帧抽取一次”的运行时策略。

因此，当前 32 帧主要来自 DeepfakeBench 预处理和 JSON 生成阶段。

## 3. 本地数据帧数范围

对 `/data/hjr/dfbench/dataset_json` 检查得到：

| 数据集 | 每个视频帧数范围 |
|---|---:|
| Celeb-DF-v2 | 3–32 |
| DeepFakeDetection | 1–32 |
| DFDC | 1–32 |
| DFDCP | 1–32 |

部分视频少于 32 帧时，代码不会补帧，模型会使用实际存在的帧数。

## 4. 代码中的潜在采样问题

代码注释表达的意图是：当原始视频帧数超过 32 时，均匀抽取 32 帧：

```python
step = total_frames // self.frame_num
frame_paths = [
    frame_paths[i]
    for i in range(0, total_frames, step)
][:self.frame_num]
```

但在计算 `step` 之前，代码已经执行：

```python
total_frames = self.frame_num
```

所以如果未来 JSON 中包含超过 32 帧，实际的：

```python
step = total_frames // self.frame_num
```

会变成 `32 // 32 = 1`，最终取前 32 个 JSON 帧，而不是按照原始视频总帧数均匀采样。

当前本地数据最大帧数就是 32，因此这个问题不会改变目前的测试结果；但如果接入包含完整视频帧的 JSON，应修正采样逻辑。

## 5. Video-level AUC 的计算

当前模型不是把 32 帧堆叠成一个视频张量进行一次预测，而是：

1. 每帧独立进行推理；
2. 获取每帧 fake probability；
3. 按视频目录聚合帧结果；
4. 对同一视频的帧概率取平均；
5. 使用视频平均概率计算 `video_auc`。

这与论文中“average the model output probabilities of each video” 的 video-level AUC 协议一致。

## 6. 结论

当前测试并不是“每隔固定 N 帧动态抽取 32 帧”，而是直接使用预处理 JSON 中已经列出的最多 32 个帧路径。需要复用 Effort 到其他数据集时，应先确认 JSON 是否已经完成 DeepfakeBench 风格的帧采样；如果 JSON 包含超过 32 帧，则应先修正 `abstract_dataset.py` 的均匀采样逻辑。

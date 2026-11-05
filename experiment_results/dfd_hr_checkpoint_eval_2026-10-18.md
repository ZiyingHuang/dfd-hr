# DFD-HR Pretrained Checkpoint Evaluation

- Checkpoint: `checkpoints/ckpt_best.pth`
- Mode: RGB (`lmdb: False`)
- RGB root: `/data/hjr/dfbench/rgb`
- Manifest root: `/data/hjr/dfbench/dataset_json`
- Sampling: 8 frames per video, resolution 448
- Evaluation date: 2026-10-18

## Official DFD-HR Table 1 Protocol-1 datasets currently available

| Dataset | Test videos | Frame AUC | Video AUC | Frame ACC | Video ACC | Frame EER | Video EER | Frame AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | 518 | 0.877536 | 0.938334 | 0.804059 | 0.853282 | 0.209302 | 0.140449 | 0.931834 |
| DeepFakeDetection | 3429 | 0.921529 | 0.952152 | 0.896402 | 0.925051 | 0.151452 | 0.137741 | 0.989897 |
| DFDC | 4704 | 0.852957 | 0.882401 | 0.746341 | 0.761267 | 0.232945 | 0.203456 | 0.871844 |
| DFDCP | 654 | 0.873998 | 0.911951 | 0.787493 | 0.834356 | 0.209491 | 0.165217 | 0.936350 |

All four Table 1 datasets completed successfully. Their manifest-referenced frames resolved without missing files, and no videos were skipped.

## Supplementary evaluations

These are not part of the Table 1 Protocol-1 comparison. FaceShifter was evaluated previously but is intentionally excluded from the one-click paper reproduction script; the other rows are additional local diagnostics:

| Dataset | Frame AUC | Video AUC | Frame ACC | Video ACC |
|---|---:|---:|---:|---:|
| FaceShifter | 0.847825 | 0.902041 | 0.773214 | 0.810714 |
| FaceForensics++ (test split) | 0.985775 | 0.999592 | 0.948393 | 0.960714 |
| Celeb-DF-v1 | 0.885870 | 0.937607 | 0.776250 | 0.770000 |
| UADFV | 0.973989 | 0.981674 | 0.792092 | 0.775510 |

The FaceForensics++ validation split was not evaluated; the current `scripts/test.py` uses the manifest `test` split.

## Official datasets not currently evaluable

The official detector configuration also lists the following datasets, but the required JSON manifests are not currently present:

```text
test_DFR, test_WDF, test_FFIW,
uniface_ff, blendface_ff, mobileswap_ff, e4s_ff,
facedancer_ff, fsgan_ff, inswap_ff, simswap_ff
```

Some raw/preprocessed frames exist for parts of these datasets elsewhere, but without matching DFD-HR JSON manifests they are not included in the official evaluation table.

## Reproduction command

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=training \
python3 scripts/test.py \
  --detector_path ./training/config/detector/dfd_hr.yaml \
  --test_dataset Celeb-DF-v2 FaceShifter DeepFakeDetection DFDC DFDCP \
  --weights_path ./checkpoints/ckpt_best.pth
```

The protobuf environment variable is required in the current environment because the installed TensorBoard/protobuf versions are incompatible. The results represent a partial reproduction of the DFD-HR paper evaluation because several official test manifests are still unavailable.

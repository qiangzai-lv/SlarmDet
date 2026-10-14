# SLARM Aggregator + PETR

Single-frame six-camera detection using the SLARM Aggregator as the encoder.
The Gaussian/motion/semantic heads, LSeg teacher and rendering operators are not
loaded. PETR's decoder, matching, ordinary detection losses and box coding are
retained. A thin PETRHead subclass handles empty 3D targets, which the original
generic sampler represents with the wrong empty box dimension.

## Run

From the SlarmDet repository root:

```bash
conda activate mmcv
python -m pip install -r projects/SLARMPETR/requirements.txt --no-deps
bash tools/dist_train.sh projects/SLARMPETR/configs/slarm_petr_singleframe.py 1
```

This starts a new randomly initialized model. Aggregator, normalization,
feature adapter and detection head are trainable. It does not resume or load
the old VoVNet checkpoint. Output is `work_dirs/slarm_petr_singleframe`.
Training uses mini train (323 frames); validation/testing use mini val (81).
Both use deterministic resizing/cropping, with no GridMask or random BEV aug.

The input dataset still produces 320x800 views. The RGB preprocessor keeps
values in [0,255]; the backbone converts to [0,1] and the Aggregator performs
ImageNet normalization once. Internally the encoder resizes the same cropped
field of view to 160x288 and scales its camera intrinsics. PETR's metadata and
position encoding retain the original 320x800 image coordinates. The default
feature shape is `[B,6,256,10,18]`. Set
`model.img_backbone.encoder_size=None` to encode at 320x800 and produce
`[B,6,256,20,50]`, after profiling the higher attention cost.

Defaults: Conv patch embedding, patch 8, embed_dim 768, depth 12, heads 16;
last-layer frame/global concatenated feature dimension 1536. This initial
integration supports Conv patch encoders only. CPU uses FP32; GPU encoder uses
BF16 autocast, while the adapter/head receive FP32 features. Start with the
default OptimWrapper; an outer `--amp` FP16 wrapper has not been validated.

## Calibration and camera order

The backbone requires six unique nuScenes cameras, named by `camera_names` or
the parent directories of `img_path`. It reorders images/calibration into
SLARM order (front-left, front, front-right, back-right, back, back-left), then
restores the original detection order. Batch elements are kept separate.

For `E_i = lidar2cam_i`, construct camera-to-local as
`F @ E_front @ inverse(E_i)`, with F mapping OpenCV axes to forward/left/up.
All times are zero (`T=1`), attention mode is full, and no streaming cache is
used. The PETR head retains `K @ E_i` projections in the detection LiDAR frame.
The transform updates actual image dimensions after crop. Unequal padded
views, negative focal lengths from flips, invalid calibration, and image
values outside the expected range fail explicitly.

## Checkpoints and fine-tuning

When weights are available, set `model.img_backbone.checkpoint` to a complete
SLARM checkpoint with `model` or `state_dict`, containing `aggregator.*` and
`aggregated_last_tokens_norm.*`. A `module.` prefix is accepted. All expected
parameter names/shapes are checked before loading. Derived ray-grid buffers
are regenerated for the selected resolution; incompatible learned weights
are never silently skipped.

Set all architecture/token switches to match the source experiment. For a
3-camera affine-token checkpoint, explicitly set `expand_affine_tokens=True`
to replicate the tokens to six cameras; this is logged and is an initialization
adaptation, not evidence of equivalent six-camera performance.

- Freeze encoder: `frozen=True` (it stays in eval mode).
- Partial fine-tuning: `frozen=False, unfreeze_last_blocks=2` trains the last
  two frame blocks, last two global blocks, and feature normalization.
- Train all: `frozen=False, unfreeze_last_blocks=None` (default).

The neck/head remain trainable in each case. For pretrained fine-tuning,
reduce `optim_wrapper.paramwise_cfg.custom_keys.img_backbone.lr_mult` to an
appropriate value such as 0.1. To resume this detector's own training state,
use the ordinary `--resume` option; this differs from encoder initialization.

## Tests

```bash
PYTHONPATH=. python tests/test_projects/test_slarm_petr.py
PYTHONPATH=. python tests/test_projects/test_petr_geometry.py
PYTHONPATH=. python projects/SLARMPETR/tools/smoke_test.py --device cuda
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  projects/SLARMPETR/tools/smoke_test.py --ddp
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python projects/SLARMPETR/tools/runner_smoke_test.py
```

The smoke tool runs one default-model GPU optimization step, or two small CPU
DDP steps, on actual nuScenes images, then predicts. It writes no checkpoints.
The unit suite checks original-encoder BF16 numerical parity, calibration,
camera permutation/batch isolation, freeze/partial gradients, strict synthetic
checkpoint loading, no dense full-attention mask, invalid inputs, real dataset
RGB preprocessing, end-to-end loss/backward, repeated prediction, official
NuScenes result formatting and empty targets.

These checks establish implementation viability, not convergence, AP gains,
multi-GPU CUDA performance, or compatibility with an unavailable real pretrained
checkpoint. Inherited PETR multi-rank loss normalization is unchanged.

## Source provenance

`slarm/` contains isolated encoding components copied from the local
`PerceptualModel-BSA/src/models` implementation with original copyright headers
and state-dict names retained. Changes are limited to package-relative imports,
removing the unrelated `src.utils.misc` dependency, respecting caller precision,
returning early for full-attention masks, and making the ray helper feature-only
(without mutating target camera metadata). No runtime `sys.path` injection or
imports of `gsplat`, `torch_scatter`, LSeg, DeepSpeed or NPU renderers are needed.

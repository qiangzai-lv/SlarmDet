# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLARM (Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes) is a feed-forward model that unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference. It learns 3D Gaussians and scene flow jointly from sparse multi-view sequences.

**Key capabilities:**
- Fast, feed-forward, streaming inference with self-supervised learning
- 3D Gaussian reconstruction + scene flow estimation
- Real-time rendering and semantic segmentation
- Supports both NVIDIA GPU (CUDA) and Ascend NPU platforms

## Platform Support

This codebase supports dual platform execution:

**GPU (NVIDIA):**
- CUDA 12.1, PyTorch 2.3.1
- Uses `gsplat` for rendering
- Set `export DEVICE_TYPE="GPU"`

**NPU (Ascend):**
- PyTorch 2.1.0 with torch-npu
- Uses `meta_gauss_render` for rendering
- Set `export DEVICE_TYPE="NPU"`
- Requires additional environment variables (see `train.sh` or README.md)

## Commands

### Training

```bash
# Set environment variables (see train.sh for full configuration)
export DEVICE_TYPE="GPU"  # or "NPU"
export DATA_ROOT=/path/to/processed/data
export PROJECT=slarm
export EXP_NAME=your_experiment

# Run training
torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} main_slarm.py \
    --project ${PROJECT} \
    --exp_name ${EXP_NAME} \
    --dataset waymo \
    --data_root $DATA_ROOT \
    --model slarm \
    [additional args...]
```

Key training arguments:
- `--batch_size`: Per-GPU batch size
- `--num_iterations`: Total training iterations
- `--ckpt_every_n_iters`: Checkpoint frequency
- `--eval_every_n_iters`: Evaluation frequency
- `--enable_depth_loss`, `--enable_flow_reg_loss`, `--enable_perceptual_loss`: Loss configurations
- `--load_from /path/to/checkpoint.pth`: Resume from checkpoint
- `--auto_resume`: Auto-resume from latest checkpoint
- `--evaluate`: Run evaluation mode

### Inference

**Offline inference:**
```bash
torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} inference.py \
    --scene_id 525 \
    --scene_start_index 0 \
    --scene_end_index 15 \
    --load_from /path/to/checkpoint.pth \
    [additional args...]
```

**Streaming inference:**
```bash
# Use inference_stream.py with --mode window_3
torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} inference_stream.py \
    --mode window_3 \
    --load_from /path/to/checkpoint.pth \
    [additional args...]
```

### Visualization

```bash
# 3D Gaussian viewer
python tools/gs_viewer.py --dir /path/to/plyfile_dir

# Point cloud viewer
python tools/pc_viewer.py --dir /path/to/plyfile_dir --fix_view --port 8080
```

### Testing

Currently no formal test suite. Validation is done through:
- Evaluation mode: Add `--evaluate` flag when running training
- Visual inspection using viewer tools
- Metrics computed during evaluation (PSNR, SSIM, scene flow metrics, semantic mIoU)

## Architecture

### Model Structure

**SLARM Model (`src/models/slarm.py`):**
- Main model class inheriting from `nn.Module`
- Integrates patch embedding, aggregator, and various heads
- Handles both 3D and 2D Gaussian splatting
- Manages rendering pipeline for both GPU and NPU

**Aggregator (`src/models/components/aggregator/aggregator.py`):**
- Registers learnable special tokens (motion, camera, sky tokens)
- Two-stage attention: frame-wise then global
- Aggregates spatiotemporal context across frames

**Component Organization:**
- `src/models/components/layers/`: Attention blocks, MLP, patch embedding, RoPE
- `src/models/components/heads/`: Camera head, DPT head, track heads
- `src/models/components/utils/`: Geometry utilities, pose encoding, rotation
- `src/models/lseg/`: LSeg integration for semantic features

### Data Pipeline

**Dataset (`src/dataset/datasets.py`):**
- `PerceptualModelDataset`: Main dataset for training
- `PerceptualModelDatasetEval`: Evaluation dataset
- `CustomConcatDataset`: Multi-dataset concatenation with sample configuration

**Data Loading:**
- Requires preprocessed data (see `docs/WAYMO.md` for Waymo dataset preparation)
- Supports multiple datasets via concatenation
- Loads depth, flow, ground plane, semantic labels

**Preprocessing:**
- `preproc/waymo_preprocess.py`: Waymo-specific preprocessing
- `preproc/waymo_download.py`: Dataset download utility
- Processed data includes: RGB, depth, flow, camera poses, semantic labels

### Losses (`src/utils/losses.py`)

Multiple loss components:
- RGB rendering loss (with optional perceptual loss via LPIPS)
- Depth loss
- Scene flow regularization loss
- Normal loss for geometric consistency
- Semantic feature loss (MSE or classification)
- Sky opacity loss

### Rendering

**GPU rendering:**
- Uses `gsplat.rendering.rasterization` or `rasterization_2dgs`

**NPU rendering:**
- Uses custom rasterizers in `src/utils/rasterizer_1112.py` or `rasterizer_1212.py`
- Version controlled via `RENDER_OP_VERSION` environment variable

## Key Workflows

### Adding a New Dataset

1. Create preprocessing script in `preproc/`
2. Add dataset configuration to `src/dataset/constants.py`
3. Update `PerceptualModelDataset` to handle new data format
4. Create scene list file in `data/dataset_scene_list/`

### Modifying Model Architecture

1. **New special token:** Update `Aggregator` class in `src/models/components/aggregator/aggregator.py`
2. **New head:** Add to `src/models/components/heads/` and integrate in `SLARM.__init__`
3. **New loss:** Add to `src/utils/losses.py` and update `compute_loss` function

### Checkpoints and Logs

- Saved to `work_dirs/<project>/<exp_name>/`
- Structure:
  - `checkpoints/`: Model checkpoints
  - `logs/`: Training logs
  - `tensorboard/`: TensorBoard logs (if enabled)

## Dependencies

**Core:**
- PyTorch (2.3.1 for GPU, 2.1.0 for NPU)
- timm, einops, jaxtyping
- torchvision

**Rendering:**
- GPU: gsplat (specific commit: 937e299)
- NPU: meta_gauss_render (custom wheel in `third_party/`)

**Semantic:**
- OpenAI CLIP (install from git)
- LSeg models (weights in `ckpts/lseg/`)

**Visualization:**
- nerfview, viser, open3d

**Third-party:**
- Depth Anything V2 in `third_party/depth_anything_v2/`

## Important Notes

1. **Platform-specific code:** Check `is_ascend_npu()` utility in `tools/__init__.py` to handle platform differences
2. **Environment variables:** Many features controlled via env vars (e.g., `FEAT_DIST`, `PROFILING`, `CONTEXT_FEAT`)
3. **LSeg weights:** Must download and place in `ckpts/lseg/` before training/inference
4. **Data preprocessing:** Separate environment recommended for data preprocessing (TensorFlow conflicts)
5. **Gradient checkpointing:** Enabled by default, disable with `--disable_grad_checkpointing`

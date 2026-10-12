import argparse
import copy
import datetime
import json
import logging
import math
import os
import time
import deepspeed
from glob import glob

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import torch_npu.npu as torch_device_lib
    print("Running on Ascend NPU.")
except ImportError:
    print("Import torch_npu failed.")
    print("Running on Nvidia GPU.")
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.utils.data

import src.utils.distributed as distributed
import src.utils.misc as misc
from engine_tools import build_model, evaluate, evaluate_flow, evaluate_semantic, visualize
from src.dataset.constants import DATASET_DICT
from src.dataset.data_utils import prepare_inputs_and_targets
from src.dataset.samplers import InfiniteSampler, NoPaddingDistributedSampler
from src.dataset.datasets import PerceptualModelDataset, PerceptualModelDatasetEval, CustomConcatDataset
from src.dataset.ads_datasets import ADSDataset, ADSDatasetEval
from src.utils.logging import MetricLogger, WandbLogger, TensorboardLogger, setup_logging
from src.utils.losses import compute_loss
from src.utils.lpips_loss import RGBLpipsLoss
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from tools.lseg_feat_extractor import LSegFeatureExtractor
if os.getenv('PROFILING'):
    if os.getenv('DEVICE_TYPE') == 'NPU':
        from tools.profiler_npu import NPUTorchProfileHook
    else:  # DEVICE_TYPE == GPU
        from tools.profiler_gpu import GPUTorchProfileHook
from src.utils.npu_mem import check_npu
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser("PerceptualModel training", add_help=False)

    # =============== Model parameters ================= #
    parser.add_argument("--model", default="slarm", type=str)
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int, help="Number of gs dimensions")
    parser.add_argument("--use_time_token", action="store_true")
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--pred_gs_conf", action="store_true")
    parser.add_argument("--enable_lifespan", action="store_true")
    parser.add_argument("--voxelize", action="store_true")
    parser.add_argument("--voxel_size", type=float, default=0.2)
    parser.add_argument("--sigmoid_rgb", action="store_true")
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["dummy", "conv"],
        default="dummy",
        help="XXX or LatentXXX",
    )
    parser.add_argument("--num_motion_tokens", default=16, type=int, help="Number of motion tokens")

    parser.add_argument("--use_pred_camera_pose", action="store_true")
    parser.add_argument("--use_pred_depth", action="store_true")

    parser.add_argument("--add_patch_plucker_embed", action="store_true")
    parser.add_argument("--add_camera_embed", action="store_true")
    parser.add_argument("--concat_plucker_embed", action="store_true")
    parser.add_argument("--use_last_token", action="store_true")

    parser.add_argument("--vggt_pretrained_weight_filepath", type=str, default="", help="filepath of vggt pre-trained model ckpts")

    parser.add_argument("--embed_dim", type=int, default=1024, help="token embedding dimension")
    parser.add_argument("--depth", type=int, default=24, help="model attention layers number")
    parser.add_argument("--num_heads", type=int, default=16, help="the number of head")
    parser.add_argument("--patch_size", type=int, default=14, help="token patchify size")
    parser.add_argument("--patch_embed", type=str, default="dinov2_vitl14_reg", help="image token patchify model")

    # model
    parser.add_argument("--enable_depth_head", action="store_true")
    parser.add_argument("--enable_camera_head", action="store_true")
    parser.add_argument("--enable_point_head", action="store_true")
    parser.add_argument("--shortcut_rgb", action="store_true")

    parser.add_argument("--use_ms3_motion", action="store_true")
    parser.add_argument("--add_angular_velocity", action="store_true")
    parser.add_argument("--max_scale", type=float, default=0.5)
    parser.add_argument("--gs_marbles", action="store_true")
    parser.add_argument("--use_2dgs", action="store_true")
    parser.add_argument("--pesudo_3dgs", action="store_true")
    parser.add_argument("--save_gaussian", action="store_true")
    parser.add_argument("--gaussian_save_path", type=str, default="output_gs")
    parser.add_argument("--save_rendered_pc", action="store_true")
    parser.add_argument("--rendered_pc_save_path", type=str, default="output_rendered_pc")
    parser.add_argument("--render_context_view", action="store_true")
    parser.add_argument("--render_context_frame_contribution", action="store_true")
    parser.add_argument("--without_feat", action="store_true", help="pred gs without semantic feature")

    parser.add_argument("--use_render_novel_view", action="store_true")

    parser.add_argument("--lseg_model_pretrained_path", type=str, default="ckpts/lseg/lseg_model_pretrained.pth", help="filepath of lseg pre-trained vit model ckpts")
    parser.add_argument("--lseg_model_scratch_path", type=str, default="ckpts/lseg/lseg_model_scratch.pth", help="filepath of lseg pre-trained scratch model ckpts")

    # =============== Losses =============== #
    parser.add_argument("--rgb_loss_coeff", type=float, default=1.0)
    parser.add_argument("--enable_context_rgb_loss", action="store_true")
    parser.add_argument("--context_rgb_loss_coeff", type=float, default=0.1)

    parser.add_argument("--enable_depth_loss", action="store_true")

    parser.add_argument("--enable_pseudo_depth_loss", action="store_true")
    parser.add_argument("--pseudo_depth_coeff", type=float, default=0.1)

    parser.add_argument("--enable_context_depth_loss", action="store_true")
    parser.add_argument("--context_depth_loss_coeff", type=float, default=0.1)
    parser.add_argument("--context_depth_loss_with_conf", action="store_true")

    parser.add_argument("--enable_context_point_loss", action="store_true")
    parser.add_argument("--context_point_loss_coeff", type=float, default=0.1)
    parser.add_argument("--context_point_loss_with_conf", action="store_true")

    parser.add_argument("--enable_feat_loss", action="store_true")
    parser.add_argument("--enable_context_feat_loss", action="store_true")
    parser.add_argument("--feat_loss_coeff", type=float, default=1.0)
    parser.add_argument(
        "--feat_loss_type",
        type=str,
        choices=["mse", "cos_dist", "cls_prob"],
        default="mse",
        help="MSE or Cosine Distance",
    )

    # Option 1: push the sky depth to a fixed value
    parser.add_argument("--enable_sky_depth_loss", action="store_true")
    parser.add_argument("--sky_depth", type=float, default=300.0)
    # Option 2: make sky gaussians transparent and use a sky token to represent sky
    parser.add_argument("--enable_sky_opacity_loss", action="store_true")
    parser.add_argument("--sky_opacity_loss_coeff", type=float, default=0.1)
    parser.add_argument("--enable_context_sky_opacity_loss", action="store_true")
    parser.add_argument("--context_sky_opacity_loss_coeff", type=float, default=0.01)

    parser.add_argument("--enable_camera_loss", action="store_true")
    parser.add_argument("--camera_loss_coeff", type=float, default=5.0)

    parser.add_argument("--context_prediction_loss_warmup_steps", type=int, default=0)

    # long lifespan regularization loss
    parser.add_argument("--enable_lifespan_reg_loss", action="store_true")
    parser.add_argument("--lifespan_reg_coeff", type=float, default=0.005)

    # flow regularization loss
    parser.add_argument("--enable_flow_reg_loss", action="store_true")
    parser.add_argument("--flow_reg_coeff", type=float, default=0.005)

    # flow loss
    parser.add_argument("--enable_flow_loss", action="store_true")
    parser.add_argument("--flow_coeff", type=float, default=0.1)
    parser.add_argument("--flow_loss_start_iter", default=10000, type=int)

    # perceptual loss
    parser.add_argument("--enable_perceptual_loss", action="store_true")
    parser.add_argument("--perceptual_weight", default=0.05, type=float, help="LPIPS weight")
    parser.add_argument("--perceptual_loss_start_iter", default=5000, type=int)
    parser.add_argument("--lpips_model", type=str, default="VGG16", choices=["VGG16", "VGG19"])

    # ============= Optimizer and LR parameters ============= #
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=8e-4, help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=5000, help="iters to warmup LR")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=3.0, help="Gradient clip")
    parser.add_argument("--disable_grad_checkpointing", action="store_true")

    parser.add_argument("--start_iteration", default=0, type=int, help="start iteration")
    parser.add_argument("--num_iterations", default=200_000, type=int, help="num of iterations")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--deepspeed_ckpt", action="store_true", help="load ckpt saved in deepspeed mode")

    # ============= Dataset parameters ============= #
    parser.add_argument("--data_root", default="./data/SLARM_data", type=str, help="dataset path")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_semantic_label", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--dataset", default=["waymo"], type=str, nargs='+', choices=DATASET_DICT.keys())
    parser.add_argument("--load_pseudo_depth", action="store_true")
    parser.add_argument("--subset_ratio", default=1.0, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    # parser.add_argument("--load_feat_types", type=str, default=None, help="The type of feat to be loaded, input in comma-separated form")  # Only LSeg
    parser.add_argument("--online_feat", action="store_true", help="online feature extraction")
    parser.add_argument("--img_norm_for_online_feat", action="store_true", help="image normalization for online feature extraction")
    parser.add_argument("--train_hour",  default=40, type=int)
    # ============= Logging ============= #
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--num_vis_samples", type=int, default=1)
    parser.add_argument("--log_every_n_iters", type=int, default=50)
    parser.add_argument("--vis_every_n_iters", type=int, default=5000)
    parser.add_argument("--ckpt_every_n_iters", type=int, default=5000)
    parser.add_argument("--eval_every_n_iters", type=int, default=50000000000)
    parser.add_argument("--total_elapsed_time", type=float, default=0.0, help="total time elapsed")
    parser.add_argument("--keep_n_ckpts", default=1, type=int)
    parser.add_argument("--profiling_name", default="profiling")

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--visualization_only", action="store_true")

    # ============= WandB ============= #
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--entity", default="YOUR_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--overwrite_wandb", action="store_true")

    # ============= TensorBoard ============= #
    parser.add_argument("--enable_tensorboard", action="store_true")

    # ============= Evaluation ============= #
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--similarity_probs_threshold", type=float, default=0.2)

    # ============= TensorBoard ============= #
    parser.add_argument("--mode", type=str, default="full",
                        help="Attention mode: 'full' = full attention, 'causal' = causal attention, "
                        "'window_N' = sliding window attention with window size N (e.g., 'window_3').")

    return parser


def add_deepspeed_args(parser):
    """Add DeepSpeed-specific arguments"""
    parser.add_argument('--deepspeed_config', type=str, default='ds_config_z2_ot.json',
                       help='DeepSpeed configuration file')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training (set by DeepSpeed)')
    return parser


def format_training_progress(
    current_step,
    total_steps,
    iterations_per_epoch,
    average_iteration_seconds,
    lr,
    total_loss,
    loss_dict,
    throughput=None,               # 新增参数
    throughput_unit="samples/s/npu", # 新增参数，单位可自定义
    now=None,
):
    """生成平台可解析的训练进度日志，不附加 logging 的 INFO 前缀。"""
    now = now or datetime.datetime.now()
    completed_steps = current_step + 1
    total_epochs = math.ceil(total_steps / iterations_per_epoch)
    current_epoch = current_step // iterations_per_epoch + 1
    current_iteration = current_step % iterations_per_epoch + 1
    remaining_steps = max(total_steps - completed_steps, 0)
    finish_time = now + datetime.timedelta(
        seconds=average_iteration_seconds * remaining_steps
    )

    date_time = now.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    estimated_finish_time = finish_time.strftime("%m.%d %H:%M")
    additional_log = [f"LR: {lr:.2e}", f"Total: {total_loss:.4f}"]
    additional_log.extend(
        f"{name}: {value:.4f}" for name, value in loss_dict.items()
    )

    base_str = (
        f"{date_time}: [train  {current_epoch}/{total_epochs}   "
        f"{current_iteration}/{iterations_per_epoch} "
        f"Fin:{estimated_finish_time}] {', '.join(additional_log)}"
    )
    if throughput is not None:
        base_str += f" | DI_throughput: {throughput:.2f} {throughput_unit}"
    return base_str


def enable_deepspeed_distributed():
    """Initialize distributed training for DeepSpeed mode

    DeepSpeed launcher sets environment variables (RANK, WORLD_SIZE, etc.)
    We need to initialize process group BEFORE data loading to get correct world_size
    """
    # Get environment variables set by DeepSpeed launcher
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch_device_lib.set_device(local_rank)

    # Initialize process group if not already initialized
    # This is needed so that distributed.get_world_size() returns correct value
    # during data loading (before deepspeed.initialize())
    if not torch.distributed.is_initialized():
        # Determine backend based on platform
        if os.environ.get("DEVICE_TYPE") == "NPU":
            backend = "hccl"
        else:
            backend = "nccl"

        # torch.distributed.init_process_group(
        #     backend=backend,
        #     rank=rank,
        #     world_size=world_size,
        # )
        deepspeed.init_distributed(dist_backend=backend)

    # Set device based on platform
    if os.environ.get("DEVICE_TYPE") == "NPU":
        torch.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)

    return local_rank

def load_model_ddp(args, model, optimizer):
    """Load checkpoint for DeepSpeed (compatible with existing format)"""
    vis_slice_id, checkpoint_loaded = 0, False

    if args.resume_from or args.auto_resume:
        if not args.resume_from:
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[DeepSpeed-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)

            # Load model state
            msg = model.load_state_dict(checkpoint["model"], strict=True)
            logger.info(f"[DeepSpeed-resume] Loaded model: {msg}")
            checkpoint_loaded = True

            # Load optimizer state
            if "optimizer" in checkpoint and "latest_step" in checkpoint:
                # msg = optimizer.load_state_dict(checkpoint["optimizer"])
                # logger.info(f"[DeepSpeed-resume] Loaded optimizer: {msg}")
                args.start_iteration = checkpoint["latest_step"] + 1

                if "vis_slice_id" in checkpoint:
                    vis_slice_id = checkpoint["vis_slice_id"] + 1

            if "latest_step" in checkpoint:
                args.start_iteration = checkpoint["latest_step"] + 1

            if "total_elapsed_time" in checkpoint:
                args.total_elapsed_time = float(checkpoint["total_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.total_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")

            del checkpoint

    if not checkpoint_loaded and args.load_from and os.path.exists(args.load_from):
        logger.info(f"Loading checkpoint from: {args.load_from}")
        checkpoint = torch.load(args.load_from, map_location='cpu')

        # Handle different checkpoint formats
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint

        try:
            msg = model.load_state_dict(checkpoint_model, strict=False)
            checkpoint_loaded = True
            logger.info(f"[DeepSpeed-init] Loaded model: {msg}")
        except Exception as e:
            logger.error(e)

            # Handle shape mismatches (same logic as misc.load_model)
            for param_name, param in model.named_parameters():
                if param_name in checkpoint_model:
                    checkpoint_param = checkpoint_model[param_name]
                    if checkpoint_param.shape != param.shape:
                        logger.info(f"Shape mismatch for {param_name}: model={param.shape}, checkpoint={checkpoint_param.shape}")
                        # Try to load with shape adaptation
                        if len(param.shape) == 1 and len(checkpoint_param.shape) == 1:
                            min_len = min(param.shape[0], checkpoint_param.shape[0])
                            param.data[:min_len] = checkpoint_param[:min_len]
                            logger.info(f"Loaded first {min_len} elements for {param_name}")
                        elif len(param.shape) == 2 and len(checkpoint_param.shape) == 2:
                            min_rows = min(param.shape[0], checkpoint_param.shape[0])
                            min_cols = min(param.shape[1], checkpoint_param.shape[1])
                            param.data[:min_rows, :min_cols] = checkpoint_param[:min_rows, :min_cols]
                            logger.info(f"Loaded top-left {min_rows}x{min_cols} block for {param_name}")
                        else:
                            logger.info(f"Could not adapt shapes for {param_name}, skipping")
                else:
                    logger.info(f"Parameter {param_name} not found in checkpoint")

            checkpoint_loaded = True
            logger.info(f"[DeepSpeed-init] Loaded model with shape adaptations")

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")

    return vis_slice_id


def load_model_deepspeed(args, model_engine, optimizer):
    """Load checkpoint for DeepSpeed (compatible with existing format)"""
    vis_slice_id, checkpoint_loaded = 0, False

    if args.resume_from or args.auto_resume:
        if not args.resume_from:
            pattern = os.path.join(args.ckpt_dir, '*')
            entries = glob(pattern)
            # 过滤出文件夹
            subdirs = [entry for entry in entries if os.path.isdir(entry)]
            # 按文件夹名的数字大小升序排序（文件夹名是纯数字）
            subdirs.sort(key=lambda x: int(os.path.basename(x)))
            if len(subdirs) > 0:
                args.resume_from = subdirs[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[DeepSpeed-resume] Resuming from: {args.resume_from}")
            _, client_state = model_engine.load_checkpoint(
                load_dir=args.ckpt_dir, 
                tag=int(args.resume_from.split('/')[-1]),
                load_module_only=False, 
                load_optimizer_states=True,
                load_lr_scheduler_states=True, 
                load_module_strict=True, 
            )
            logger.info(f"[DeepSpeed-resume] Loaded model: Success")
            checkpoint_loaded = True

            args.start_iteration = int(args.resume_from.split('/')[-1]) + 1

    if not checkpoint_loaded and args.load_from and os.path.exists(args.load_from):
        logger.info(f"Loading checkpoint from: {args.load_from}")
        _, client_state = model_engine.load_checkpoint(
                load_dir=os.path.dirname(args.load_from), 
                tag=int(args.load_from.split('/')[-1]),
                load_module_only=False, 
                load_optimizer_states=True,
                load_lr_scheduler_states=True, 
                load_module_strict=True, 
            )
        checkpoint_loaded = True
        logger.info(f"[DeepSpeed-init] Loaded model: Success")

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")

    return vis_slice_id


def main_deepspeed(args):
    # Prepare distributed training
    local_rank = enable_deepspeed_distributed()

    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    world_size, global_rank = distributed.get_world_size(), distributed.get_global_rank()
    print("="*50,"world_size:", world_size, "global_rank:", global_rank, "="*50)
    seed = args.seed + global_rank
    misc.fix_random_seeds(seed)

    log_writer = None
    if global_rank == 0:
        for d in [log_dir, checkpoint_dir, video_dir]:
            os.makedirs(d, exist_ok=True)
        if args.enable_wandb:
            run_id_path, run_id = os.path.join(log_dir, "wandb_run_id.txt"), None
            if os.path.exists(run_id_path) and not args.overwrite_wandb:
                with open(run_id_path, "r") as f:
                    run_id = f.readlines()[-1].strip()
            log_writer = WandbLogger(args=args, resume="must", id=run_id)
            if run_id is None:
                with open(run_id_path, "a") as f:
                    f.write(log_writer.run_id + "\n")

        if args.enable_tensorboard: # NOTE: enable_tensorboard will overwrite log_writer, tensorboard and wandb cannot be used simultaneously
            log_writer = TensorboardLogger(args=args)

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("SLARM")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
    if global_rank == 0:
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)


    train_datasets = []
    val_datasets = []
    eval_datasets = []
    eval_flow_datasets = []
    dataset_names_list = []

    has_val_annotation = False

    for dataset_name in args.dataset:
        dataset_meta = DATASET_DICT[dataset_name]
        train_annotation = dataset_meta["annotation_txt_file_train"]
        val_annotation = dataset_meta["annotation_txt_file_val"]

        if train_annotation is not None:
            if dataset_name == "nuscenes":
                train_annotation = f"data/dataset_scene_list/nuscenes_train.txt"
            else:
                if dataset_name == "ads":
                    train_annotation = train_annotation
                else:
                    train_annotation = f"{args.data_root}/{train_annotation}"

        if dataset_name == "ads":
            train_datasets.append(ADSDataset(
                data_root=args.data_root,
                annotation_txt_file_list=train_annotation,
                target_size=args.input_size,
                num_context_timesteps=args.num_context_timesteps,
                num_target_timesteps=args.num_target_timesteps,
                timespan=args.timespan,
                num_max_cams=args.num_max_cameras,
                load_depth=args.load_depth,
                load_flow=args.load_flow,
                load_pseudo_depth=args.load_pseudo_depth,
                load_semantic_label=args.load_semantic_label,
                skip_sky_mask=args.skip_sky_mask,
                # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
                online_feat=args.online_feat,
                img_norm_for_online_feat=args.img_norm_for_online_feat,
                train_hour=args.train_hour
            ))
        else:
            train_datasets.append(PerceptualModelDataset(
                data_root=args.data_root,
                annotation_txt_file_list=train_annotation,
                target_size=args.input_size,
                num_context_timesteps=args.num_context_timesteps,
                num_target_timesteps=args.num_target_timesteps,
                timespan=args.timespan,
                num_max_cams=args.num_max_cameras,
                load_depth=args.load_depth,
                load_flow=args.load_flow,
                load_pseudo_depth=args.load_pseudo_depth,
                load_semantic_label=args.load_semantic_label,
                skip_sky_mask=args.skip_sky_mask,
                # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
                online_feat=args.online_feat,
                img_norm_for_online_feat=args.img_norm_for_online_feat,
            ))
        dataset_names_list.append(dataset_name)

        if val_annotation is not None:
            if dataset_name == "nuscenes":
                val_annotation = f"data/dataset_scene_list/nuscenes_val.txt"
            else:
                if dataset_name == "ads":
                    val_annotation = val_annotation
                else:
                    val_annotation = f"{args.data_root}/{val_annotation}"
            if not os.path.exists(val_annotation) and dataset_name != "ads":
                val_annotation = None
            else:
                has_val_annotation = True

        if val_annotation is not None:
            # dataloader for visualization
            EvalCls = ADSDatasetEval if dataset_name == "ads" else PerceptualModelDatasetEval
            ValCls = ADSDataset if dataset_name == "ads" else PerceptualModelDataset
            val_datasets.append(ValCls(
                data_root=args.data_root,
                annotation_txt_file_list=val_annotation,
                target_size=args.input_size,
                num_context_timesteps=args.num_context_timesteps,
                num_target_timesteps=args.num_target_timesteps,
                timespan=args.timespan,
                num_max_cams=args.num_max_cameras,
                load_depth=args.load_depth,
                load_flow=args.load_flow,
                load_semantic_label=args.load_semantic_label,
                skip_sky_mask=args.skip_sky_mask,
                # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
                online_feat=args.online_feat,
                img_norm_for_online_feat=args.img_norm_for_online_feat,
            ))
            # dataloader for evaluation (reconstruction and semantic)
            eval_datasets.append(EvalCls(
                data_root=args.data_root,
                annotation_txt_file_list=val_annotation,
                target_size=args.input_size,
                num_context_timesteps=args.num_context_timesteps,
                num_target_timesteps=args.num_target_timesteps,
                timespan=args.timespan,
                num_max_cams=args.num_max_cameras,
                load_depth=args.load_depth,
                load_flow=args.load_flow,
                load_dynamic_mask=True,
                load_ground_label=args.load_ground,
                load_semantic_label=args.load_semantic_label,
                skip_sky_mask=args.skip_sky_mask,
                # load_feat_types=args.load_feat_types.split(',') if args.load_feat_types else [],
                # online_feat=args.online_feat,
                # img_norm_for_online_feat=args.img_norm_for_online_feat,
            ))
            # dataloader for evaluation (scene flow)
            eval_flow_datasets.append(EvalCls(
                data_root=args.data_root,
                annotation_txt_file_list=val_annotation,
                target_size=args.input_size,
                num_context_timesteps=args.num_context_timesteps,
                num_target_timesteps=args.num_target_timesteps,
                timespan=args.timespan,
                num_max_cams=args.num_max_cameras,
                load_depth=args.load_depth,
                load_flow=args.load_flow,
                load_dynamic_mask=False,
                load_ground_label=args.load_ground,
                load_semantic_label=args.load_semantic_label,
                return_context_as_target=False,
                skip_sky_mask=args.skip_sky_mask,
            ))

        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"train: {train_annotation}, val: {val_annotation}")

    DATASET_SAMPLE_NUMS = json.loads(os.environ.get("DATASET_SAMPLE_NUMS", "null"))

    if len(train_datasets) > 1:
        dataset_train = CustomConcatDataset(train_datasets, dataset_names=dataset_names_list, per_dataset_sample_nums=DATASET_SAMPLE_NUMS)
    else:
        dataset_train = train_datasets[0] if train_datasets else None

    if len(val_datasets) > 1:
        dataset_val = CustomConcatDataset(val_datasets, dataset_names=dataset_names_list)
    else:
        dataset_val = val_datasets[0] if val_datasets else None

    if len(eval_datasets) > 1:
        dataset_eval = CustomConcatDataset(eval_datasets, dataset_names=dataset_names_list)
    else:
        dataset_eval = eval_datasets[0] if eval_datasets else None

    if len(eval_flow_datasets) > 1:
        dataset_eval_flow = CustomConcatDataset(eval_flow_datasets, dataset_names=dataset_names_list)
    else:
        dataset_eval_flow = eval_flow_datasets[0] if eval_flow_datasets else None

    sampler_train = InfiniteSampler(
        sample_count=len(dataset_train) * distributed.get_world_size(),
        shuffle=True,
        seed=seed
    )
    print("="*50, "sample_count：", len(dataset_train) * torch.distributed.get_world_size(), "="*50)
    print("="*50, "torch.distributed.get_world_size()：", torch.distributed.get_world_size(), "="*50)

    # dataloader for training and visualization
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )

    # InfiniteSampler 在每个 rank 上提供完整数据量，据此定义一个 epoch。
    iterations_per_epoch = max(1, len(dataset_train) // args.batch_size)

    if has_val_annotation:
        # sampler = NoPaddingDistributedSampler(
        #     dataset_eval,
        #     num_replicas=world_size,
        #     rank=global_rank,
        #     shuffle=False,
        # )
        sampler = DistributedSampler(
            dataset_eval,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
            drop_last=True,  # 自动 pad，32张卡都严格执行相同的步数
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )
        data_loader_eval_flow = torch.utils.data.DataLoader(
            dataset_eval_flow,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )
    else:
        dataset_val = None
        dataset_eval = None
        dataset_eval_flow = None
        data_loader_eval = None
        data_loader_eval_flow = None

    if DATASET_SAMPLE_NUMS is not None:
        logger.info(f"Specified per-dataset sample sizes: {DATASET_SAMPLE_NUMS}")
    logger.info(f"Dataset {args.dataset} contains {len(dataset_train):,} training sequences in total.")

    model = build_model(args)

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model_without_ddp = model

    # Load DeepSpeed configuration
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    logger.info(f"DeepSpeed config: {ds_config}")

    # Prepare optimizer (same as main_slarm.py)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))


    if not args.deepspeed_ckpt:
        print("============================Load ckpt saved in ddp mode.=====================================")
        vis_slice_id = load_model_ddp(args, model_without_ddp, optimizer)


    # Initialize DeepSpeed (pass optimizer, not model_parameters)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    # model_engine is the DeepSpeed engine
    model_without_ddp = model_engine.module  # Unwrapped model

    logger.info(f"DeepSpeed engine initialized")
    logger.info(f"Optimizer = {optimizer}")

    # Load checkpoint or resume training
    logger.info(f"Original start Iteration: {args.start_iteration}")
    if args.deepspeed_ckpt:
        print("============================Load ckpt saved in deepspeed mode.=====================================")
        vis_slice_id = load_model_deepspeed(args, model_engine, optimizer)
    logger.info(f"New start iteration {args.start_iteration}")

    global_batch_size = args.batch_size * world_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * global_batch_size / 256
    logger.info("Global batch size: %d" % global_batch_size)
    logger.info(f"Base lr: {args.lr * 256 / global_batch_size:.2e}, Actual lr: {args.lr:.2e}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    logger.info(f"Training with {world_size} GPUs")

    # Load feature extractor
    # if args.online_feat and args.load_feat_types is not None:
    if args.online_feat:
        assert args.enable_feat_loss or args.enable_context_feat_loss
        logger.info("Using online feature, loading feature extractor.")
        feat_extractor = LSegFeatureExtractor(args.lseg_model_pretrained_path, args.lseg_model_scratch_path, dtype=torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16)
        logger.info("Feature extractor loaded.")
    else:
        feat_extractor = None

    data_iter_step = args.start_iteration
    if log_writer is not None:
        log_writer.set_step(data_iter_step)

    dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16

    if args.evaluate:
        torch.distributed.barrier()         
        eval_result = evaluate(data_loader_eval, model_without_ddp, args)
        if log_writer is not None and eval_result is not None:
            eval_result = {f"eval/{k}": v for k, v in eval_result.items()}
            log_writer.update(eval_result)
        if "waymo" in args.dataset or "driving_sim" in args.dataset:
            if args.decoder_type != "conv":
                flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
                if log_writer is not None and flow_eval_result is not None:
                    flow_eval_result = {f"eval/{k}": v for k, v in flow_eval_result.items()}
                    log_writer.update(flow_eval_result)
        # if args.load_semantic_label and args.load_feat_types is not None and "waymo" in args.dataset:
        if args.load_semantic_label:
            if "waymo" in args.dataset or "b2d" in args.dataset or "driving_sim" in args.dataset:
                semantic_eval_result = evaluate_semantic(data_loader_eval, model_without_ddp, args)
                if log_writer is not None and semantic_eval_result is not None:
                    semantic_eval_result = {f"eval/{k}": v for k, v in semantic_eval_result.items()}
                    log_writer.update(semantic_eval_result)
        logger.info("Evaluation done, exiting.")
        exit()

    valid_slice_id = copy.deepcopy(vis_slice_id)
    if dataset_val is not None and valid_slice_id >= len(dataset_val):
        valid_slice_id = 0
    # if distributed.is_main_process():
    #     for _ in range(args.num_vis_samples):
    #         vis_slice_id, valid_slice_id = visualize(
    #             args=args,
    #             model=model_without_ddp,
    #             dset_train=dataset_train,
    #             step=data_iter_step,
    #             train_vis_id=vis_slice_id,
    #             device=device,
    #             dset_val=dataset_val,
    #             val_vis_id=valid_slice_id,
    #             log_writer=log_writer,
    #             feat_extractor=feat_extractor,
    #         )
    if args.visualization_only:
        logger.info("Visualization done, exiting.")
        exit()

    rgb_and_lpips_loss = RGBLpipsLoss(
        perceptual_weight=args.perceptual_weight,
        enable_perceptual_loss=args.enable_perceptual_loss,
        lpips_model=args.lpips_model,
    ).to(device)
    # will turn on perceptual loss after a certain number of iterations
    rgb_and_lpips_loss.set_perceptual_loss(False)

    logger.info(f"Starting training from iteration {args.start_iteration} to {args.num_iterations}")
    metrics_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    start_time = time.time()
    # ---------- 新增吞吐量计时器 ----------
    throughput_start_time = start_time                # 起始时刻
    throughput_start_step = args.start_iteration      # 起始步数（已完成的步数）
    # ------------------------------------
    num_tokens_printed = False

    if os.getenv('PROFILING'):
        if os.getenv('DEVICE_TYPE') == 'NPU':
            profiler = NPUTorchProfileHook(args.profiling_name, skip_first=5, wait=5, warmup=10, active=1, repeat=1, with_stack=False)
        else:  # DEVICE_TYPE == GPU
            profiler = GPUTorchProfileHook(args.profiling_name, skip_first=5, wait=5, warmup=10, active=1, repeat=1, with_stack=False)
        profiler.before_train()
    # Print timing for steps 20, 40, 60, 80, 100 for debugging time consumption
    time_count_step = [20, 40, 60, 80, 100] if os.getenv("TIME_COUNT_TYPE1") else [-1]
    s0 = time.time()  # Timer initialization for time debugging
    global_timer = time.time()
    step_timer = time.time()
    for data_dict in metric_logger.log_every(
        data_loader_train,
        print_freq=args.log_every_n_iters,
        header="Training",
        n_iterations=args.num_iterations,
        start_iteration=args.start_iteration,
    ):
        print("#" * 30, "data_iter_step begin:", data_iter_step, "#" * 30)
        if data_iter_step <= 1:
            global_timer = time.time()
        elif data_iter_step <= 10:
            print("step time:", time.time() - step_timer, "avg time:", (time.time() - global_timer) / (data_iter_step - 1))
            check_npu("start step", debug=True)
        step_timer = time.time()
        check_npu("before epoch")
        if data_iter_step in time_count_step:  # time debug
            print('##################')
            print("data time:", (time.time() - s0) * 1000)
            s0 = time.time()
        if data_iter_step > args.num_iterations:
            break
        if log_writer is not None:
            log_writer.set_step(data_iter_step)
        if args.enable_perceptual_loss and data_iter_step >= args.perceptual_loss_start_iter:
            rgb_and_lpips_loss.set_perceptual_loss(True)

        model_engine.train()
        misc.adjust_learning_rate(optimizer, data_iter_step, args)

        with torch.autocast("cuda", dtype=dtype):
            if not isinstance(data_dict['num_max_cams'], int):
                num_max_cams = int(data_dict['num_max_cams'][0])
                data_dict["fps"] = float(data_dict["fps"][0])
            else:
                num_max_cams = data_dict['num_max_cams']
            input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams,
                                                                feat_extractor=feat_extractor)
            if data_iter_step in time_count_step:  # time debug
                print('##################')
                print("data preprocess time:", (time.time() - s0) * 1000)
                s0 = time.time()
            if os.environ.get('TIME_COUNT_TYPE2'):
                start = time.time()
            # Forward pass (DeepSpeed handles mixed precision)
            pred_dict = model_engine(input_dict)
            check_npu("after forward")

            if data_iter_step in time_count_step:  # time debug
                print('##################')
                print("forward time:", (time.time() - s0) * 1000)
                s0 = time.time()

            # print('='*50, "global_rank:", global_rank, "pred_dict:", pred_dict["render_results"]["rendered_feat"].shape, "="*50)
            
            loss_dict = compute_loss(pred_dict, input_dict, target_dict, args, rgb_and_lpips_loss, data_iter_step)
            check_npu("after loss")
            if data_iter_step in time_count_step:  # time debug
                print('##################')
                print("compute loss time:", (time.time() - s0) * 1000)
                s0 = time.time()

        loss_value = sum(loss for k, loss in loss_dict.items() if "loss" in k)

        if not math.isfinite(loss_value):
            logger.info("NaN detected")
            raise AssertionError

        if os.environ.get('TIME_COUNT_TYPE2'):
             
            backward_start = time.time()

        # DeepSpeed backward + step (handles gradient scaling and clipping)
        model_engine.backward(loss_value)
        check_npu("after backward")
        model_engine.step()
        check_npu("after step")

        # Get gradient norm from DeepSpeed
        grad_norm = model_engine.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = 0.0

        if data_iter_step in time_count_step:  # time debug
            print('##################')
            print("backward time:", (time.time() - s0) * 1000)
            s0 = time.time()

        if os.environ.get('TIME_COUNT_TYPE2'):
            print('Backward pass:',  time.time() - backward_start)
            print('End-to-end time:',  time.time() - start)

        if world_size > 1:
            loss_dict = {k: v.float() for k, v in loss_dict.items()}  # avoid overflow during allreduce
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / world_size for k, v in loss_dict.items()}
        total_loss_reduced = sum(loss for k, loss in loss_dict_reduced.items() if "loss" in k)
        lr = optimizer.param_groups[0]["lr"]
        psnr = -10 * np.log10(loss_dict_reduced["rgb_mse"]) if "rgb_mse" in loss_dict_reduced else 0
        metric_logger.update(lr=lr, psnr=psnr, loss=total_loss_reduced, **loss_dict_reduced)
        metric_logger.update(grad_norm=grad_norm)

        if distributed.is_main_process() and (
            data_iter_step % args.log_every_n_iters == 0
            or data_iter_step + 1 >= args.num_iterations
        ):
            # ---------- 计算吞吐量（新增） ----------
            current_time = time.time()
            steps_since = data_iter_step - throughput_start_step + 1   # 包含当前步
            time_since = current_time - throughput_start_time
            if time_since > 0 and steps_since > 0:
                # 单卡吞吐量 = batch_size / 每步平均耗时 = batch_size * steps_since / time_since
                throughput = args.batch_size * steps_since / time_since
            else:
                throughput = 0.0
            # 重置计时器，为下一个打印区间做准备
            throughput_start_time = current_time
            throughput_start_step = data_iter_step + 1
            # ------------------------------------

            completed_steps_in_run = data_iter_step - args.start_iteration + 1
            average_iteration_seconds = (
                time.time() - start_time
            ) / max(completed_steps_in_run, 1)
            print(
                format_training_progress(
                    current_step=data_iter_step,
                    total_steps=args.num_iterations,
                    iterations_per_epoch=iterations_per_epoch,
                    average_iteration_seconds=average_iteration_seconds,
                    lr=lr,
                    total_loss=total_loss_reduced,
                    loss_dict=loss_dict_reduced,
                    throughput=throughput,                # 传入计算值
                    throughput_unit="samples/s/npu",      # 可根据需要改为 "tokens/s/npu"
                ),
                flush=True,
            )

        if "num_tokens" in pred_dict and not num_tokens_printed:
            logger.info(f"num_tokens: {pred_dict['num_tokens']}")
            num_tokens_printed = True

        if log_writer is not None:
            train_result = {
                    "psnr": psnr,
                    "loss": total_loss_reduced,
                    **loss_dict_reduced,
                    "lr": lr,
                    "grad_norm": grad_norm,
                }
            train_result = {f"train/{k}": v for k, v in train_result.items()}
            log_writer.update(train_result)
            log_writer.set_step()

        if (data_iter_step + 1) % args.ckpt_every_n_iters == 0:
            if not args.deepspeed_ckpt:
                if distributed.is_main_process():
                    elapsed_t = time.time() - start_time + args.total_elapsed_time
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "latest_step": data_iter_step,
                        "vis_slice_id": vis_slice_id,
                        "args": args,
                        "total_elapsed_time": elapsed_t,
                        "deepspeed": True,
                    }
                    checkpoint_path = os.path.join(args.ckpt_dir, f"ckpt_{data_iter_step:06d}.pth")
                    torch.save(checkpoint, checkpoint_path)
                    misc.cleanup_checkpoints(args.ckpt_dir, keep_num=args.keep_n_ckpts)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                model_engine.save_checkpoint(args.ckpt_dir, data_iter_step, client_state = {'step': data_iter_step})
                # 手动添加 DeepSpeed 分片目录的清理逻辑
                if distributed.is_main_process():
                    # 获取所有以数字命名的子目录（DeepSpeed 保存的 tag）
                    subdirs = [d for d in os.listdir(args.ckpt_dir) if os.path.isdir(os.path.join(args.ckpt_dir, d)) and d.isdigit()]
                    subdirs.sort(key=int)  # 按步数排序
                    torch.cuda.empty_cache()
                    # 只保留最新的 keep_n_ckpts 个
                    while len(subdirs) > args.keep_n_ckpts:
                        dir_to_remove = os.path.join(args.ckpt_dir, subdirs.pop(0))
                        import shutil
                        logger.info(f"Removed old checkpoint directory: {dir_to_remove}")
                        shutil.rmtree(dir_to_remove)

        if (data_iter_step + 1) % args.vis_every_n_iters == 0:
            if distributed.is_main_process():
                for _ in range(args.num_vis_samples):
                    vis_slice_id, valid_slice_id = visualize(
                        args=args,
                        model=model_without_ddp,
                        dset_train=dataset_train,
                        step=data_iter_step,
                        train_vis_id=vis_slice_id,
                        device=device,
                        dset_val=dataset_val,
                        val_vis_id=valid_slice_id,
                        log_writer=log_writer,
                        feat_extractor=feat_extractor,
                    )
            torch.distributed.barrier()
            torch.cuda.empty_cache()

        if (data_iter_step + 1) % args.eval_every_n_iters == 0 and (
            data_iter_step + 1
        ) != args.num_iterations:
            eval_result = evaluate(data_loader_eval, model_without_ddp, args, f"{data_iter_step}")
            if log_writer is not None and eval_result is not None:
                log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})
            if args.decoder_type != "conv":
                if "waymo" in args.dataset or "driving_sim" in args.dataset:
                    flow_eval_result = evaluate_flow(
                        data_loader_eval_flow,
                        model_without_ddp,
                        args,
                        name_str=f"{data_iter_step}",
                    )
                    if log_writer is not None and flow_eval_result is not None:
                        log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})
            # if args.load_semantic_label and args.load_feat_types is not None and "waymo" in args.dataset:
            if args.load_semantic_label:
                if "waymo" in args.dataset or "b2d" in args.dataset or "driving_sim" in args.dataset:
                    semantic_eval_result = evaluate_semantic(
                        data_loader_eval,
                        model_without_ddp,
                        args,
                        name_str=f"{data_iter_step}",
                    )
                    if log_writer is not None and semantic_eval_result is not None:
                        log_writer.update({f"eval/{k}": v for k, v in semantic_eval_result.items()})
            torch.distributed.barrier()
            torch.cuda.empty_cache()

        if data_iter_step in time_count_step:  # time debug
             
            print('##################')
            print("postprocess time:", (time.time() - s0) * 1000)
            if data_iter_step == time_count_step[-1]:
                break
        data_iter_step += 1
        if data_iter_step in time_count_step:  # time debug
             
            s0 = time.time()
        if os.getenv('PROFILING'):
            profiler.after_step()

        del pred_dict
        del loss_dict
        del loss_value

    if os.getenv('PROFILING'):
        profiler.after_train()

    metric_logger.synchronize_between_processes()

    total_time = time.time() - start_time + args.total_elapsed_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    # eval_result = evaluate(data_loader_eval, model_without_ddp, args)
    # if log_writer is not None and eval_result is not None:
    #     log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})
    # if args.decoder_type != "conv":
    #     if "waymo" in args.dataset or "driving_sim" in args.dataset:
    #         flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
    #         if log_writer is not None and flow_eval_result is not None:
    #             log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})
    # # if args.load_semantic_label and args.load_feat_types is not None and args.dataset == "waymo":
    # if args.load_semantic_label:
    #     if "waymo" in args.dataset or "b2d" in args.dataset or "driving_sim" in args.dataset:
    #         semantic_eval_result = evaluate_semantic(data_loader_eval, model_without_ddp, args)
    #         if log_writer is not None and semantic_eval_result is not None:
    #             log_writer.update({f"eval/{k}": v for k, v in semantic_eval_result.items()})
    logger.info("Done!")
    if log_writer is not None:
        log_writer.finish()


if __name__ == "__main__":
    if os.getenv("DETERMINISTIC"):
        torch.use_deterministic_algorithms(True)
    if os.getenv("DEBUG_AUTOGRAD"):
        torch.autograd.set_detect_anomaly(True)
    # Parse arguments
    parser = get_args_parser()
    # 关键步骤：添加DeepSpeed的命令行参数
    parser = deepspeed.add_config_arguments(parser)
    # parser = add_deepspeed_args(parser)
    args = parser.parse_args()

    # Run DeepSpeed training
    main_deepspeed(args)
#!/bin/bash


# Select platform
export DEVICE_TYPE="NPU"  # GPU or NPU

if [ "$DEVICE_TYPE" = "GPU" ]; then
    # run single-GPU inference demo
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_DEBUG=INFO
    export NCCL_P2P_DISABLE=0
    export NCCL_P2P_LEVEL=NVL
fi

if [ "$DEVICE_TYPE" = "NPU" ]; then
    export DEVICE_TYPE="NPU"
    export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
    # NPU performance optimization
    export AVOID_AI_CPU=1
    export USE_EQUAL_CROSS=1
    export TASK_QUEUE_ENABLE=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export CONTEXT_FEAT=1
    export RENDER_OP_VERSION=1212
    # export ASCEND_LAUNCH_BLOCKING=1

    # Replace the rendering preprocessing script in the container environment
    bash replace_meta_gauss_render.sh ${CONDA_DEFAULT_ENV} ${RENDER_OP_VERSION}
fi

# Set model configuration
export FEAT_DIST=1
export DATASET=waymo
export DATA_ROOT=/efs-large-spatial-model-2/waymo_storm_processed/STORM_data
# export OVERFIT_EXP=1
# export SCENE_ID_WAYMO=80
# export PROFILING=1  # Printing takes time

export MASTER_PORT=20798
export DEVICE_NUM=2
export BS_PER_DEVICE=1
export PROJECT=slarm
export EXP_NAME=exp_debug_ds

# export CKPT_PTH=ckpts/ckpt_199999.pth     # 精度有问题
# export CKPT_PTH=work_dirs/slarm/exp_0529_192-256_lastest/checkpoints/ckpt_002499.pth   # 在ckpt_199999.pth上finetune
# export CKPT_PTH=ckpts/ckpt_003999.pth  # 志成提供：阶段1 3DGS+特征蒸馏，阶段2 语义label监督
# export CKPT_PTH=/efs-mi-pay-gy1/c00595937/SLARM_2/PerceptualModel/work_dirs/slarm_finetune_labelloss/exp_0602/checkpoints/ckpt_009999.pth  # 志成提供：阶段1 3DGS+特征蒸馏，阶段2 语义label监督
# export CKPT_PTH=work_dirs/slarm/exp_0616_160-240/checkpoints/ckpt_037499.pth   # 160*240, dim=1536, depth=24, num_heads=16


# python -m debugpy --listen 13688 --wait-for-client train_ds.py \
# --auto_resume, --load_from $CKPT_PTH, --evaluate
# deepspeed --num_gpus=$DEVICE_NUM --master_port ${MASTER_PORT} train_ds.py \
torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} train_ds.py \
    --project ${PROJECT} \
    --exp_name ${EXP_NAME} \
    --dataset ${DATASET} \
    --data_root $DATA_ROOT \
    --batch_size $BS_PER_DEVICE --num_iterations 200000 --lr_sched constant \
    --vis_every_n_iters 500 \
    --eval_every_n_iters 2500 --keep_n_ckpts 30 --ckpt_every_n_iters 2500 \
    --log_every_n_iters 1 \
    --enable_tensorboard \
    --model slarm \
    --load_depth --load_flow --load_ground \
    --load_semantic_label \
    --num_max_cameras 3 --use_affine_token \
    --sigmoid_rgb \
    --num_motion_tokens 0 \
    --use_sky_token \
    --embed_dim 2048 --depth 12 --num_heads 16 --patch_embed conv --patch_size 8 \
    --use_ms3_motion \
    --use_last_token \
    --shortcut_rgb \
    --add_patch_plucker_embed \
    --enable_depth_loss --enable_sky_opacity_loss \
    --enable_flow_reg_loss --flow_reg_coeff 0.005 \
    --enable_perceptual_loss --perceptual_weight 0.05 --perceptual_loss_start_iter 5000 \
    --rgb_loss_coeff 1.0 \
    --similarity_probs_threshold 0.2 \
    --online_feat --img_norm_for_online_feat \
    $( [ "$DEVICE_TYPE" = "GPU" ] && echo "--enable_feat_loss" ) \
    $( [ "$DEVICE_TYPE" = "NPU" ] && echo "--enable_context_feat_loss" ) \
    --feat_loss_coeff 1.0 --feat_loss_type mse \
    --lseg_model_scratch_path ckpts/lseg/lseg_model_scratch.pth --lseg_model_pretrained_path ckpts/lseg/lseg_model_pretrained_2.pth \
    --input_size 256 384 \
    --deepspeed \
    --deepspeed_config ds_config_z2_ot.json \
    --deepspeed_ckpt \
    # --without_feat \
    # --load_from $CKPT_PTH \
    # --evaluate \
    # --auto_resume
    # --online_feat --img_norm_for_online_feat \
    # $( [ "$DEVICE_TYPE" = "GPU" ] && echo "--enable_feat_loss" ) \
    # $( [ "$DEVICE_TYPE" = "NPU" ] && echo "--enable_context_feat_loss" ) \
    # --feat_loss_coeff 1.0 --feat_loss_type mse \
    # --lseg_model_scratch_path ckpts/lseg/lseg_model_scratch.pth --lseg_model_pretrained_path ckpts/lseg/lseg_model_pretrained_2.pth \
    # --load_from $CKPT_PTH \
    # --evaluate \
    # --evaluate \
    # --deepspeed_ckpt \

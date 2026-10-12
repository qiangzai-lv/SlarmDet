#!/bin/bash


DEVICE_TYPE="NPU"

# 将代码和数据都放在 efs-mi-east4-1 中，云上只需要启动该脚本即可
# 代码路径
WORKDIR="/efs-mi-pay-gy1/l00800213/vit_optimize/PerceptualModel/"

# 环境变量设置
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=29512

# 打印出参数的值
echo "NNODES: ${MA_NUM_HOSTS}"
echo "MASTER_HOST: ${VC_WORKER_HOSTS}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NGPUS_PER_NODE: ${MA_NUM_GPUS}"
echo "NUM_PROCESSES: $((${MA_NUM_GPUS} * ${MA_NUM_HOSTS}))"

CONDA_ENV="render_lta2"
RENDER_OP_VERSION=1212

# step1: 需将本sh脚本的所有目录都chmod 777，比如 /efs-mi-pay-gy1/c00595937/PerceptualModel/training_roma.sh
# step2: 若未软链接，将下行代码置于 sudo -i bash -i -c 所包含的命令中，也可以手动先进行软链接

# # 数据路径
# SOURCE_DIR="/efs-large-spatial-model-2/waymo_storm_processed/STORM_data"
# # 数据在代码路径中的软链接目录
# TARGET_DIR="data/STORM_data"
# ln -s "${SOURCE_DIR}" "${TARGET_DIR}" && \
    # export OVERFIT_EXP=1 && \
    # export SCENE_ID=089 && \
    # export RENDER_PROJ_FWD_USE_FUSED_KERNEL=1 && \

DATASET=waymo
PROJECT=slarm
EXP_NAME=exp_0529_192-256
DATA_ROOT=/efs-large-spatial-model-2/waymo_storm_processed/STORM_data
BS_PER_DEVICE=2

sudo -i bash -i -c "
    export HCCL_CONNECT_TIMEOUT=6000 && \
    cd ${WORKDIR} && \
    conda activate ${CONDA_ENV} && \
    export AVOID_AI_CPU=1 && \
    export USE_EQUAL_CROSS=1 && \
    export TASK_QUEUE_ENABLE=2 && \
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True && \
    export FEAT_DIST=1 && \
    export CONTEXT_FEAT=1 && \
    export RENDER_OP_VERSION=${RENDER_OP_VERSION} && \
    bash replace_meta_gauss_render.sh ${CONDA_ENV} ${RENDER_OP_VERSION} && \
    bash cp_clip_weight.sh && \
    torchrun --nnodes=${MA_NUM_HOSTS} --nproc_per_node=${MA_NUM_GPUS} --node_rank=${VC_TASK_INDEX} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} main_slarm.py \
        --project ${PROJECT} \
        --exp_name ${EXP_NAME} \
        --dataset ${DATASET} \
        --data_root $DATA_ROOT \
        --batch_size $BS_PER_DEVICE --num_iterations 200000 --lr_sched constant \
        --vis_every_n_iters 500 \
        --eval_every_n_iters 10000 --keep_n_ckpts 30 --ckpt_every_n_iters 10000 \
        --enable_tensorboard \
        --model slarm \
        --load_depth --load_flow --load_ground \
        --load_semantic_label \
        --num_max_cameras 3 --use_affine_token \
        --sigmoid_rgb \
        --num_motion_tokens 0 \
        --use_sky_token \
        --embed_dim 768 --depth 12 --patch_embed conv --patch_size 8 \
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
        --load_from ckpts/ckpt_199999.pth \
        --input_size 192 256 \
        --disable_grad_checkpointing > ${EXP_NAME}.txt
"

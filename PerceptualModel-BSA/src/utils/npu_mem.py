import torch
import torch_npu
import src.utils.distributed as distributed
# import logging
# import sys

# logger = logging.getLogger("SLARM")

# 设置默认使用的 NPU 设备 (例如 NPU 0)
def check_npu(prompt="", debug=False):
    if not debug:
        return # 正式训练关闭check
    rank = distributed.get_global_rank()
    if rank != 0:
        return
    device = torch.device("npu")
    # 1. 查看当前已分配的显存 (单位：Byte，通常需要转成 MB 或 GB)
    allocated_memory = torch.npu.memory_allocated(device)
    reserved_memory = torch.npu.memory_reserved(device)
    max_allocated = torch.npu.max_memory_allocated(device)
    # logger.info(f"\033[32m***DEVICE [{prompt}] 当前已分配显存: {allocated_memory / 1024**2:.2f} MB 保留显存: {reserved_memory / 1024**2:.2f} MB 显存占用峰值: {max_allocated / 1024**2:.2f} MB\033[0m")
    # logger.info(f"***DEVICE [{prompt}] 当前已分配显存: {allocated_memory / 1024**2:.2f} MB 保留显存: {reserved_memory / 1024**2:.2f} MB 显存占用峰值: {max_allocated / 1024**2:.2f} MB")
    print(f"***DEVICE [{prompt}] 当前已分配显存: {allocated_memory / 1024**2:.2f} MB 保留显存: {reserved_memory / 1024**2:.2f} MB 显存占用峰值: {max_allocated / 1024**2:.2f} MB")

def check_tensor(t, prompt="", debug=False):
    if not debug:
        return # 正式训练关闭check
    rank = distributed.get_global_rank()
    if rank != 0:
        return
    mem_bytes = t.numel() * t.element_size()
    mem_mb = mem_bytes / (1024 ** 2)
    # logger.info(f"***TENSOR [{prompt}] shape={t.shape} dtype={t.dtype} mem={mem_mb:.2f} MB")
    print(f"***TENSOR [{prompt}] shape={t.shape} dtype={t.dtype} mem={mem_mb:.2f} MB")
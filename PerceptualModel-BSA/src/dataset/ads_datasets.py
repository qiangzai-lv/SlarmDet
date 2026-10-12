import base64
import gzip
import io
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple
import cv2

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
import torchvision.transforms as transforms

from .constants import DATASET_DICT, DATASETS, MEAN, STD
from .datasets import PerceptualModelDataset
from .data_utils import resize_depth, to_tensor, to_float_tensor, depth2xyz
from moxing.framework.util import runtime
runtime.reset_logger(level=logging.ERROR)


def _inject_ads_fields(scene_json: Dict) -> None:
    """
    将 ads scene.json 补齐父类 get_segment 需要的字段。
    原地修改 scene_json dict，不写回文件。
    """
    scene_json["dataset"] = "ads"
    scene_json["scene_name"] = scene_json["scene_id"]
    samples = scene_json.get("samples", [])
    scene_json["num_timesteps"] = len(samples)
    # 从 timestamp 推算 fps 和 normalized_time
    timestamps = []
    for s in samples:
        try:
            t = float(s.get("timestamp", 0))
        except (ValueError, TypeError):
            t = None
        if t is not None:
            timestamps.append(t)
    if len(timestamps) > 1:
        fps = 1.0 / (timestamps[1] - timestamps[0])
    else:
        fps = 10.0
    scene_json["fps"] = fps
    if timestamps:
        t0 = timestamps[0]
        scene_json["normalized_time"] = [t - t0 for t in timestamps]
    else:
        scene_json["normalized_time"] = []

logger = logging.getLogger("PerceptualModel")


class ADSDataset(PerceptualModelDataset):
    """
    商用车队感知数据集适配器。
    继承 PerceptualModelDataset，只重写 get_frame() 方法，
    复用父类的 get_segment()、get_one_scene() 时序逻辑。

    annotation_txt_file_list 中的 .shrink 文件每行是 base64+gzip+pickle 编码的 JSON。
    """

    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: str,
        target_size: Tuple[int, int] = (160, 240),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[3] = 3,
        timespan: float = 2.0,  # 2.0 seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_pseudo_depth: bool = False,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        load_semantic_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        feat_input_size: Tuple[int, int] = (320, 480),
        online_feat: bool = False,
        img_norm_for_online_feat: bool = False,
        only_interp: bool = True,
        train_hour: int = 40,
    ):
        # 用空文件骗过父类 __init__（我们自己在下面解析 shrink）
        tmp_txt = os.path.join("./", ".ads_placeholder.txt")
        with open(tmp_txt, "w") as f:
            f.write("")

        super().__init__(
            data_root=data_root,
            annotation_txt_file_list=tmp_txt,
            target_size=target_size,
            num_context_timesteps=num_context_timesteps,
            num_target_timesteps=num_target_timesteps,
            num_max_cams=num_max_cams,
            timespan=timespan,
            subset_indices=subset_indices,
            num_replicas=num_replicas,
            equispaced=equispaced,
            load_depth=load_depth,
            load_pseudo_depth=load_pseudo_depth,
            load_flow=load_flow,
            load_dynamic_mask=load_dynamic_mask,
            load_ground_label=load_ground_label,
            load_semantic_label=load_semantic_label,
            return_context_as_target=return_context_as_target,
            skip_sky_mask=skip_sky_mask,
            feat_input_size=feat_input_size,
            online_feat=online_feat,
            img_norm_for_online_feat=img_norm_for_online_feat,
            only_interp=only_interp,
        )
        # ------------------------------
        # 用子类自己的 shrink 解析替代父类的 annotation 加载
        # ------------------------------
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]

        self.annotations = []
        for txt_file in annotation_txt_file_list:
            self._load_shrink_txt(txt_file, train_hour)

        logger.info(f"ADS shrink loading complete: {len(self.annotations)} scene(s) loaded. train hout : {train_hour}")

        # 给每个 scene.json 注入字段，让父类的 get_segment 能正常工作
        for scene_json in self.annotations:
            _inject_ads_fields(scene_json)

        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                # transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        # original size transformation for online feature extraction
        if img_norm_for_online_feat:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),  # Larger deviation from offline features, TODO: determine whether to do img norm based on implementation results
            ]
        else:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
            ]
        self.img_to_extract_feat_transformation = transforms.Compose(img_to_extract_feat_transformation_list)

        self.only_interp = only_interp

    def _load_shrink_txt(self, shrink_obs_path: str, traindata_total_hour: int) -> None:
        """
        加载 .shrink 格式的 txt 文件。
        每行是 base64+gzip+pickle 编码的 JSON scene 对象。（不是文件路径）
        """
        temp = []
        shrink_obs_path = "/home/ma-user/modelarts/user-job-dir/code/D3P_data_43200_DATASET_common_haisi_0728-v151-release-unknow_type-1af83cf5fe_VERSION_202607291619_train_43182_18970081_guiyang_r_pkg0_10frames.jsonl.shrink"
        if shrink_obs_path.startswith("obs"):
            import moxing as mox
            with mox.file.File(shrink_obs_path, "rb") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if idx > 960:
                        compressed_data = base64.b64decode(line)
                        pickled_data = gzip.decompress(compressed_data)
                        raw_json_str = pickle.loads(pickled_data)
                        scene = json.loads(raw_json_str)
                        self.annotations.append(scene)       # 存入训练集
        else:
            with open(shrink_obs_path, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if (idx + 1) * 30 / 3600 < traindata_total_hour:
                        compressed_data = base64.b64decode(line)
                        pickled_data = gzip.decompress(compressed_data)
                        raw_json_str = pickle.loads(pickled_data)
                        scene = json.loads(raw_json_str)

                        # 获取 samples 列表，避免 None
                        samples = scene.get("samples", [])

                        # 检查是否所有样本的 depth_cam1 字段都不以 ".npy" 结尾
                        # 注意：如果某个样本没有 depth_cam1 字段，我们视为不以 ".npy" 结尾（因为 get 返回 None，用空字符串替代）
                        all_not_npy = all(
                            not ((sample.get("depth_cam1") or "").endswith(".npy") or (sample.get("img_cam1") or "").endswith(".jpg") )
                            for sample in samples
                        )

                        if all_not_npy:
                            self.annotations.append(scene)
                        else:
                            temp.append(scene)

                # 循环结束后打印统计信息
                logger.info(f"use {len(self.annotations)} train , vaild is {len(temp)}")
                     


    # ------------------------------------------------------------------
    #  OBS / parquet 读取工具
    # ------------------------------------------------------------------


    def _read_parquet(self, obs_path: str) -> Optional[pd.DataFrame]:
        import moxing as mox
        if not obs_path or not isinstance(obs_path, str) or len(obs_path) < 4:
            logger.warning(f"[_read_parquet] invalid path: {obs_path}")
            return None
        try:
            df = pd.read_parquet(mox.file.File(obs_path, "rb"))
            return df
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"[_read_parquet] path={obs_path}  mox read failed: {e}")
        # fallback: 本地路径
        try:
            df = pd.read_parquet(obs_path)
            logger.info(f"[_read_parquet] fallback SUCCESS path={obs_path} shape={df.shape}")
            return df
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"[_read_parquet] path={obs_path} fallback also failed: {e}")
            return None


    # ------------------------------------------------------------------
    #  外参 / 内参 构建
    # ------------------------------------------------------------------

    def _get_vehicle_to_world(self, utils_data) -> np.ndarray:
        """
        从 sample 的 utils.json 读取 ego_pose.odom_pose，
        构建 4×4 vehicle→world 变换矩阵。
        """
        ego_pose = utils_data.get("ego_pose", {})
        odom_pose = ego_pose.get("odom_pose", None)
        if odom_pose is None:
            raise ValueError(f"no odom_pose in utils: {utils_path}")

        # odom_pose[0] = [qw, qx, qy, qz]
        # odom_pose[1] = [x, y, z]
        qw, qx, qy, qz = odom_pose[0]
        tx, ty, tz = odom_pose[1]

        rot = R.from_quat([qx, qy, qz, qw])  # scipy 用 [x, y, z, w]
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot.as_matrix().astype(np.float32)
        mat[:3, 3] = [tx, ty, tz]
        return mat

    def _get_intrinsic(self, calib: Dict, camera: str) -> np.ndarray:
        """
        获取相机内参并按 target_size 缩放。
        原始尺寸为 calib 中 width/height。
        """
        info = calib.get(camera)
        if info is None:
            return np.eye(3, dtype=np.float32)

        K = np.array(info.get("camera_intrinsic", np.eye(3)), dtype=np.float32).copy()
        orig_w = float(info.get("width", 3840.0))
        orig_h = float(info.get("height", 2160.0))
        scale_w = self.target_size[1] / orig_w
        scale_h = self.target_size[0] / orig_h
        K[0] *= scale_w
        K[1] *= scale_h
        K[0, 2] *= scale_w
        K[1, 2] *= scale_h
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        return K, fx, fy, cx, cy

    def _get_sky_mask(self, img_np: np.ndarray) -> np.ndarray:
        """
        从图像提取天空掩码。简单做法：判断蓝色通道显著高于红绿通道的像素。
        如果 ads 数据有单独的 sky_mask 字段，应优先使用独立 mask.
        """
        if img_np is None:
            img_hsv = None
        else:
            from skimage.color import rgb2hsv
            hsv = rgb2hsv(img_np[:, :, :3])
            img_hsv = hsv
        # 示例：s 低、v 高的像素可能是天空, 但需要根据实际数据的 sky mask 调整
        return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)

    def _get_cam_install(self, calib, camera: str) -> np.ndarray:
        """从 calib json 构建相机安装外参（相机→车身 变换）。"""
        info = calib.get(camera)
        if info is None:
            raise ValueError(f"calib missing camera: {camera}")

        R_mat = np.array(info.get("rotation", np.eye(3)), dtype=np.float32)
        t = np.array(info.get("translation", np.zeros(3)), dtype=np.float32).reshape(3)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R_mat
        mat[:3, 3] = t
        return mat

    def _load_depth_map(self, df, camera_name):
        if camera_name not in df['names'].values:
            return None
        idx = df['names'].tolist().index(camera_name)
        depth_bytes = df['datas'].iloc[idx]
        depth = pickle.loads(depth_bytes)
        return depth

    # ------------------------------------------------------------------
    #  get_frame() 重写
    # ------------------------------------------------------------------

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        重写父类的 get_frame 方法。
        从 scene_json 的 samples 数组读取第 frame_idx 个 sample，
        加载 parquet 数据，返回与原始 get_frame 兼容的结构。
        """
        import moxing as mox
        dataset_name = scene_json["dataset"]
        samples = scene_json["samples"]

        sample = samples[frame_idx]
        calib_obs_path = sample.get("calib", {})
        calib = json.load(mox.file.File(calib_obs_path, "rb"))

        # ---- 读取当前帧的 vehicle_to_world（从 utils.json） ----
        utils_obs_path = sample.get("utils", "")
        utils_data = json.load(mox.file.File(utils_obs_path, "rb"))
        vehicle_to_world = self._get_vehicle_to_world(utils_data)

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]

        # 构建 ref_camera 的世界到规范化的变换
        ref_sample = samples[source_frame_idx]
        ref_utils_path = ref_sample.get("utils", "")
        ref_utils_data = json.load(mox.file.File(ref_utils_path, "rb"))
        ref_vehicle_to_world = self._get_vehicle_to_world(ref_utils_data)
        ref_cam_install = self._get_cam_install(calib, ref_camera_name)
        ref_cam_to_world = ref_vehicle_to_world @ ref_cam_install
        world_to_canonical = np.linalg.inv(ref_cam_to_world)

        images = []
        depths = []
        camtoworlds = []
        intrinsics = []
        pts3ds = []
        valid_masks = []
        sky_masks = []
        images_to_extract_feat = []  # using for online feature extraction


        img_obs_key = "img_cam1"
        depth_obs_key = "depth_cam1"
        img_obs_path = sample.get(img_obs_key, "")
        df_img = self._read_parquet(img_obs_path)
        depth_obs_path = sample.get(depth_obs_key, "")
        df = self._read_parquet(depth_obs_path)

        for camera in camera_list:
            # ---- 图像 ----
            jpeg_bytes = df_img[camera].iloc[0]
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            pil_img = Image.open(io.BytesIO(nparr.tobytes()))
            img = self.img_transformation(pil_img)
            images.append(img)

            # ---- 内参 ----
            K, fx, fy, cx, cy = self._get_intrinsic(calib, camera)
            intrinsics.append(torch.from_numpy(K).float())

            # ---- 外参 ----
            cam_install = self._get_cam_install(calib, camera)
            cam_to_world = vehicle_to_world @ cam_install
            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ cam_to_world
                @ DATASETS[dataset_name]["opencv2dataset"]
            )
            camtoworlds.append(to_tensor(camtoworld))

            if self.online_feat:
                img_to_extract_feat = self.img_to_extract_feat_transformation(pil_img)
                images_to_extract_feat.append(img_to_extract_feat)


            # ---- 深度 ----
            depth_np = None
            if self.load_depth:
                depth = torch.tensor(self._load_depth_map(df, camera))
                depth = resize_depth(depth, self.target_size)
                depths.append(depth)

                # pts3d and valid_mask
                pts3d = depth2xyz(np.array(depth),
                                fxfycxcy=np.array([fx / self.target_size[1], fy / self.target_size[0], cx / self.target_size[1], cy / self.target_size[0]]),
                                cam2world=np.array(camtoworld),
                                return_pixel=True)
                pts3d = torch.tensor(pts3d).float()
                pts3ds.append(pts3d)
                valid_mask = (depth > 0.0) & (depth < 200)
                valid_masks.append(valid_mask)

            sky = torch.zeros(self.target_size[0], self.target_size[1]).float()
            sky = to_tensor(np.array(sky) > 0).float()
            sky_masks.append(sky)

        # ---- 组装 frame_out ----
        frame_out = {}
        if self.online_feat:  # online features extraction
            frame_images_to_extract_feat = torch.stack(images_to_extract_feat)
            frame_out["frame_images_to_extract_feat"] = frame_images_to_extract_feat
        if len(images) > 0:
            frame_out["image"] = torch.stack(images, dim=0)
        if len(camtoworlds) > 0:
            frame_out["camtoworld"] = torch.stack(camtoworlds, dim=0)
        if len(intrinsics) > 0:
            frame_out["intrinsics"] = torch.stack(intrinsics, dim=0)
        if len(depths) > 0:
            frame_out["depth"] = torch.stack(depths, dim=0)
        if len(pts3ds) > 0:
            frame_out["pts3d"] = torch.stack(pts3ds, dim=0)
        if len(valid_masks) > 0:
            frame_out["valid_masks"] = torch.stack(valid_masks, dim=0)
        if len(sky_masks) > 0:
            frame_out["sky_masks"] = torch.stack(sky_masks, dim=0)
        frame_out["frame_idx"] = frame_idx

        return to_float_tensor(frame_out)


class ADSDatasetEval(ADSDataset):
    """
    ADSDataset 的 eval 版本。
    只加载 shrink 文件中前 960 个不以 .npy/.jpg 结尾的场景（与训练时验证集筛选条件一致）。
    """
    def _load_shrink_txt(self, shrink_obs_path: str, traindata_total_hour: int) -> None:
        shrink_obs_path = "/home/ma-user/modelarts/user-job-dir/code/WEN4800clips_datas_2.shrink"
        if shrink_obs_path.startswith("obs"):
            import moxing as mox
            with mox.file.File(shrink_obs_path, "rb") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if idx < 960:
                        compressed_data = base64.b64decode(line)
                        pickled_data = gzip.decompress(compressed_data)
                        raw_json_str = pickle.loads(pickled_data)
                        scene = json.loads(raw_json_str)
                        samples = scene.get("samples", [])
                        all_not_npy = all(
                            not ((sample.get("depth_cam1") or "").endswith(".npy")
                                 or (sample.get("img_cam1") or "").endswith(".jpg"))
                            for sample in samples
                        )
                        if all_not_npy:
                            self.annotations.append(scene)
                    else:
                        break
        else:
            with open(shrink_obs_path, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    compressed_data = base64.b64decode(line)
                    pickled_data = gzip.decompress(compressed_data)
                    raw_json_str = pickle.loads(pickled_data)
                    scene = json.loads(raw_json_str)
                    samples = scene.get("samples", [])
                    all_not_npy = all(
                        not ((sample.get("depth_cam1") or "").endswith(".npy")
                                or (sample.get("img_cam1") or "").endswith(".jpg"))
                        for sample in samples
                    )
                    if all_not_npy:
                        self.annotations.append(scene)


        logger.info(f"ADSDatasetEval: loaded {len(self.annotations)} eval scene(s).")



# ============================================================
# 测试 main 函数
# ============================================================
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ADSDataset.test")

    if len(sys.argv) < 2:
        print("Usage: python -m src.dataset.ads_dataset <shrink_file> [--num-cams N]")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Test ADSDataset")
    parser.add_argument("shrink_file", type=str, help="Path to .shrink file")
    parser.add_argument("--num-cams", type=int, default=3, help="Number of cameras (default: 3)")
    parser.add_argument("--load-depth", action="store_true", default=True, help="Load depth data")
    parser.add_argument("--target-size", type=int, nargs=2, default=(160, 240), metavar=("H", "W"), help="Target image size (default: 160 240)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ADSDataset 测试开始")
    logger.info(f"shrink_file: {args.shrink_file}")
    logger.info(f"num_cams:   {args.num_cams}")
    logger.info(f"load_depth: {args.load_depth}")
    logger.info(f"target_size: {args.target_size}")

    # --------------------------------------------------------
    # 1. 加载 .shrink 文件，解码得到第一个 scene_json
    # --------------------------------------------------------
    shrink_path = Path(args.shrink_file)
    if not shrink_path.exists():
        logger.error(f"shrink file not found: {shrink_path}")
        sys.exit(1)

    # 逐行读取，只解码第一行（测试用的场景）
    first_line = None
    with open(shrink_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break

    if first_line is None:
        logger.error("shrink file is empty!")
        sys.exit(1)

    compressed_data = base64.b64decode(first_line)
    pickled_data = gzip.decompress(compressed_data)
    raw_json_str = pickle.loads(pickled_data)
    scene_json = json.loads(raw_json_str)
    _inject_ads_fields(scene_json)

    logger.info(f"scene_id:     {scene_json.get('scene_id', 'N/A')}")
    logger.info(f"vehicle_name: {scene_json.get('vehicle_name', 'N/A')}")
    logger.info(f"dataset:      {scene_json.get('dataset', 'N/A')}")
    logger.info(f"samples count: {len(scene_json.get('samples', []))}")

    # --------------------------------------------------------
    # 2. 检查 calib
    # --------------------------------------------------------
    calib = scene_json.get("calib", {})
    if not calib:
        logger.warning("calib is empty in scene.json!")
    else:
        logger.info(f"calib cameras: {list(calib.keys())}")
        for cam_name in list(calib.keys())[:3]:
            info = calib[cam_name]
            logger.info(
                f"  {cam_name}: "
                f"{info.get('width', '?')}x{info.get('height', '?')}, "
                f"K={info.get('camera_intrinsic', 'N/A')}"
            )

    # --------------------------------------------------------
    # 3. 检查 samples 的 OBS 路径
    # --------------------------------------------------------
    samples = scene_json.get("samples", [])
    if not samples:
        logger.error("No samples in scene.json!")
        sys.exit(1)

    sample0 = samples[0]
    logger.info(f"\nSample[0] keys: {list(sample0.keys())}")
    for key in ["img_cam1", "depth_cam1", "pv_seg", "calib", "utils"]:
        val = sample0.get(key, "")
        if val:
            logger.info(f"  {key}: {str(val)[:120]}...")
        else:
            logger.info(f"  {key}: (not present)")

    # --------------------------------------------------------
    # 4. 实例化 ADSDataset
    # --------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("实例化 ADSDataset ...")

    try:
        dataset = ADSDataset(
            data_root=str(shrink_path.parent),
            annotation_txt_file_list=str(shrink_path.resolve()),
            num_max_cams=args.num_cams,
            target_size=args.target_size,
            load_depth=args.load_depth,
        )
    except Exception as e:
        logger.error(f"实例化失败: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"ADSDataset 实例化成功")

    # --------------------------------------------------------
    # 5. 测试 get_frame
    # --------------------------------------------------------
    test_indices = [0]
    if len(samples) > 30:
        test_indices.append(30)
    if len(samples) > 1:
        test_indices.append(len(samples) - 1)

    for frame_idx in test_indices:
        logger.info(f"\n--- get_frame(frame_idx={frame_idx}) ---")
        try:
            frame_out = dataset.get_frame(scene_json, frame_idx)
        except Exception as e:
            logger.error(f"get_frame({frame_idx}) 失败: {e}", exc_info=True)
            continue

        logger.info(f"frame_out keys: {sorted(frame_out.keys())}")
        for key in frame_out:
            val = frame_out[key]
            if isinstance(val, torch.Tensor):
                logger.info(
                    f"  {key}: shape={list(val.shape)}, "
                    f"dtype={val.dtype}, "
                    f"range=[{val.min().item():.4f}, {val.max().item():.4f}]"
                )
            elif isinstance(val, list):
                logger.info(f"  {key}: len={len(val)}, type={type(val[0]).__name__ if val else 'empty'}")
            else:
                logger.info(f"  {key}: {type(val).__name__}")

        # 检查图像数据
        if "image" in frame_out:
            img = frame_out["image"]
            logger.info(f"  image valid: {img.abs().sum().item():.2f}, "
                        f"NaN count: {torch.isnan(img).sum().item()}")

        # 检查深度数据
        if "depth" in frame_out:
            depth = frame_out["depth"]
            valid_pixels = (depth > 1e-6).sum().item()
            total_pixels = depth.numel()
            logger.info(
                f"  depth valid pixels: {valid_pixels}/{total_pixels} "
                f"({100.0 * valid_pixels / max(total_pixels, 1):.1f}%)"
            )

    # --------------------------------------------------------
    # 6. 数据完整性摘要
    # --------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("测试摘要")
    if 'frame_out' in dir():
        for frame_idx in test_indices[:1]:
            logger.info(f"  Frame {frame_idx}: {sorted(frame_out.keys())}")
    logger.info("ADSDataset 测试完成")



class ADSSingleSequenceDataset(ADSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        scene_json = self.annotations[index % len(self.annotations)]
        num_timesteps = scene_json["num_timesteps"]
        if end_index < 0 or end_index > num_timesteps:
            end_index = num_timesteps
        segment_data = []
        cam_to_world = scene_json["camera_to_world"]
        ref_camera_name = DATASET_DICT[scene_json["dataset"]]["ref_camera"]

        # world_to_canonical
        world_to_canonical = np.linalg.inv(torch.tensor(cam_to_world[ref_camera_name][start_index]))  # dtype('float32')

        clip_start_id = None
        interval = self.get_interval(scene_json['fps'])
        for start_id in trange(start_index, end_index, interval):
            # Handle the last segment in the scene that does not meet the interval (this has already been inferred in the previous few frames).
            if start_id + interval > num_timesteps:
                clip_start_id = start_id
                start_id = num_timesteps - interval
                print(f' ---- Total num_timesteps is {num_timesteps}, the last segment start at {start_id}, will clip at {clip_start_id}. ---- ')

            segment_dict = self.get_segment(
                index=index, context_frame_idx=start_id, return_all=True
            )
            # compute the relative transformation from a start_id to the first frame
            current_world = cam_to_world[ref_camera_name][start_id]

            # canonical (flu)
            segment_to_ref = (DATASETS[scene_json['dataset']]["canonical_to_flu"] @ world_to_canonical) @ \
                                (current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"]))
            # worlds (flu)
            # segment_to_ref = DATASETS[scene_json['dataset']]["canonical_to_flu"] @ \
            #                     current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"])

            # worlds (opencv)
            # opencv2waymo = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            # segment_to_ref = np.linalg.inv(opencv2waymo) @ DATASETS[scene_json['dataset']]["canonical_to_flu"] @ \
            #                     current_world @ np.linalg.inv(DATASETS[scene_json['dataset']]["canonical_to_flu"])
            # print('segment_to_ref: Point cloud in world coordinate system!')

            segment_dict["segment_to_ref"] = to_float_tensor(segment_to_ref)
            if clip_start_id is not None:
                segment_dict["relative_clip_start_id"] = clip_start_id - start_id
            segment_data.append(segment_dict)
        return segment_data
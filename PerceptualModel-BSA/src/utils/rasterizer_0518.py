# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import math

from typing import Dict, Optional, Tuple
from typing_extensions import Literal
import torch
from torch import Tensor

import acl
from meta_gauss_render import (
    spherical_harmonics,
    # projection_three_dims_gaussian_fused,
    flash_gaussian_build_mask,
    gaussian_sort,
    calc_render,
    get_render_schedule,
)
from .projection_three_dims_gaussian_fused_1212 import projection_three_dims_gaussian_fused


def assert_check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(
    gs: Tuple,
    viewmats: Tensor,
    Ks: Tensor,
    sh_degree: Optional[int],
    N: int,
    C: int,
    camera_model: str = "pinhole",
    render_mode: str = "RGB+ED"
) -> None:
    means, quats, scales, opacities, colors = gs
    assert_check(means.shape == (N, 3), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}")
    assert_check(Ks.shape == (C, 3, 3), f"Invalid shape for Ks: {Ks.shape}")
    assert_check(camera_model in ["pinhole", "ortho", "fisheye"], f"Invalid camera_model: {camera_model}")
    assert_check(render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}")

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [ N, D] or [C, N, D]
        assert_check((colors.dim() == 2 and colors.shape[0] == N) or 
                    (colors.dim() == 3 and colors.shape[:2] == (C, N)),
                    f"Invalid shape for colors: {colors.shape}")
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert_check((colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or
                    (colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3),
                    f"Invalid shape for colors: {colors.shape}")

        assert_check((sh_degree + 1) ** 2 <= colors.shape[-2], f"Invalid sh_degree for colors shape: {colors.shape}")


class Rasterizer:
    def __init__(self, *args, **kwargs) -> None:
        self.tile_size = 32
        self.pix_coord = None
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.tilenum_height = None
        self.tilenum_width = None

    def tile2image(self, rendered_image, height, width, channel_dim=3):
        ts = self.tile_size
        nh = self.tilenum_height
        nw = self.tilenum_width
        rendered_image = rendered_image.reshape(nh, nw, ts, ts, -1).transpose(1, 2).reshape(nh * ts, nw * ts, -1)
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def get_render_input(self, gs, depths, _cam_view):
        means2d, colors, opacities, conics = gs
        cf_means2 = means2d[0, _cam_view]
        cf_colors3 = colors[0, _cam_view]
        cf_opacity = opacities[0, _cam_view]

        inv_x_0 = conics[0, _cam_view, 0, :]
        inv_x_1 = conics[0, _cam_view, 1, :]
        inv_x_2 = conics[0, _cam_view, 2, :]

        cf_depths = depths[_cam_view]
        ts = self.tile_size
        height = self.padded_height
        width = self.padded_width
        pix_coords = self.pix_coord.reshape(height // ts, ts, width // ts, ts, 2) \
            .permute(0, 2, 1, 3, 4).reshape(height // ts * width // ts, ts * ts, 2) \
            .permute(0, 2, 1).to(torch.float32).contiguous()
        cf_gs = cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths 
        return (cf_gs, pix_coords)

    def ascend_rasterize_splats(
        self,
        viewmats: Tensor,
        Ks: Tensor,
        size: Tuple,
        tile_size: int,
        active_sh_degree: int,
        splats: dict,
    ) -> Tuple[Tensor, Tensor, Dict]:
        width, height = size
        if self.tile_grid is None:
            self.tile_size = tile_size
            self.tilenum_height = math.ceil(height / tile_size)
            self.tilenum_width = math.ceil(width / tile_size)
            self.padded_height = self.tilenum_height * tile_size
            self.padded_width = self.tilenum_width * tile_size
            device = splats["means"].device
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size), \
                torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), 
                torch.arange(self.padded_height), indexing='xy'), dim=-1).to(device)

        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        scales = splats["scales"]   # torch.exp(splats["scales"])  # [N, 3]
        opacities = splats["opacities"]  # torch.sigmoid(splats["opacities"])  # [N,]

        colors = splats["colors"]  # torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
        flow = None
        if colors.shape[-1] == 6:  # split rgn and flow
            flow = colors[:, 3:]
            colors = colors[:, :3]

        render_colors, render_depth, render_alpha, info = self._ascend_rasterization(
            gs=(means, quats, scales, opacities, colors),
            viewmats=viewmats,
            Ks=Ks,
            size=size,
            sh_degree=active_sh_degree,
            tile_size=tile_size,
            camera_model="pinhole",
            render_mode="RGB+ED",
        )
        if flow is not None:
            render_flows, _, _, _ = self._ascend_rasterization(
                gs=(means, quats, scales, opacities, flow),
                viewmats=viewmats,
                Ks=Ks,
                size=size,
                sh_degree=active_sh_degree,
                tile_size=tile_size,
                camera_model="pinhole",
                render_mode="RGB+ED",
            )
            render_colors = torch.cat((render_colors, render_flows, render_depth), -1)
        else: render_colors = torch.cat((render_colors, render_depth), -1)
        return render_colors, render_alpha, info

    def _ascend_rasterization(
        self,
        gs: Tuple[Tensor],
        viewmats: Tensor,
        Ks: Tensor,
        size: Tuple,
        sh_degree: Optional[int] = None,
        tile_size: int = 32,
        camera_model: str = "pinhole",
        render_mode: str = "RGB+ED"
    ) -> Tuple[Tensor, Tensor, Dict]:
        means, quats, scales, opacities, colors = gs

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        validate_inputs(gs, viewmats, Ks, sh_degree, N, C, camera_model, render_mode)
        width, height = size

        colors = colors[:, None]
        if sh_degree is None:
            # Colors are post-activation values, with shape [N, D] or [C, N, D]
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
        else:
            # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
            camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)
            else:
                # colors is already [C, N, K, 3]
                shs = colors

            # build colors
            rays_o = camtoworlds[0, :3, 3]
            rays_d = means - rays_o
            rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
            k = (sh_degree + 1) ** 2
            colors = spherical_harmonics(sh_degree, rays_d.reshape(B, N, 3), shs[0, :, :k, :].reshape(B, N, k, 3))  
            colors = (colors + 0.5).clip(min=0.0)

        # NOTE: when sh_degree is not None, shape of colors is (1, 3, N), when sh_degree is None, shape of colors is (N, 1, 3)
        if sh_degree is None:
            '''
            colors shape is inconsistent when sh_degree is None vs non-None
            colors shape in projection_three_dims_gaussian_fused needs to match the shape when sh_degree is non-None
            otherwise it will cause rendering accuracy issues
            '''
            colors = colors.permute(1, 2, 0).contiguous()

        # ascend gauss projection
        means2d, depths, conics, opacities, radius, covars2d, colors, cnt = projection_three_dims_gaussian_fused(
            means.reshape(B, N, 3),
            colors,
            None,
            quats.reshape(B, N, 4),
            scales.reshape(B, N, 3),
            opacities.reshape(B, N),
            viewmats.reshape(B, C, 4, 4).contiguous(),
            Ks.reshape(B, C, 3, 3),
            width,
            height,
            0.3,
            0.2
        )
        camera_ids, gaussian_ids = None, None

        # ascend gauss sort
        with torch.no_grad():
            tile_sums, tile_offsets, tile_depths, tile_gauss_ids = (
                flash_gaussian_build_mask(
                    means2d,
                    opacities[None, :],
                    conics,
                    covars2d,
                    depths,
                    cnt[None, :],
                    self.tile_grid.float(),
                    width,
                    height,
                    tile_size,
                )
            )
            sorted_cnts = tile_offsets.squeeze(-1)[:, :, -1]  # [B, C]
            sorted_cnts_flatten = sorted_cnts.flatten()
            sorted_offset = torch.cumsum(sorted_cnts_flatten, dim=0)
            tile_sums_3d = tile_sums.squeeze(-1)  # [B, C, T]
            tile_sums_cpu = tile_sums_3d.cpu().to(torch.int64)

            vector_num = acl.get_device_capability(0, 1)[0]
            lb_sched_tensor_cpu = get_render_schedule(tile_sums_cpu, vector_num)
            lb_sched_tensor = lb_sched_tensor_cpu.npu()
            max_tile_gauss = torch.amax(tile_sums).item()

            sorted_gs_ids = gaussian_sort(
                lb_sched_tensor,
                tile_sums,
                tile_depths,
                tile_gauss_ids,
                sorted_offset,
                max_tile_gauss,
            )

        render_colors = []
        render_depths = []
        render_alphas = []
        for _batch_id in range(0, B):
            for _cam_view in range(0, C):
                gs = means2d, colors, opacities, conics
                cf_gs, pix_coords = self.get_render_input(gs, depths, _cam_view)
                (
                    cf_means2,
                    cf_colors3,
                    cf_opacity,
                    inv_x_0,
                    inv_x_1,
                    inv_x_2,
                    cf_depths,
                ) = cf_gs
                view_idx = _batch_id * C + _cam_view
                start_index = sorted_offset[view_idx - 1] if view_idx > 0 else 0
                end_index = sorted_offset[view_idx]
                sorted_gs_id = sorted_gs_ids[start_index:end_index]
                lb_sched = lb_sched_tensor[_batch_id, _cam_view, :]
                # ascend rasterize to pixels
                cf_render_colors, cf_render_depths = calc_render(
                    cf_means2,
                    inv_x_0,
                    inv_x_1,
                    inv_x_2,
                    cf_opacity,
                    cf_colors3,
                    cf_depths,
                    pix_coords,
                    lb_sched,
                    sorted_gs_id,
                )
                cf_render_colors = self.tile2image(
                    cf_render_colors.permute(1, 2, 0), height, width
                )
                cf_render_depths = self.tile2image(
                    cf_render_depths.permute(1, 2, 0), height, width
                )

                cf_colors3_for_alphas = torch.ones_like(cf_colors3)
                cf_render_alphas, _ = calc_render(
                    cf_means2,
                    inv_x_0,
                    inv_x_1,
                    inv_x_2,
                    cf_opacity,
                    cf_colors3_for_alphas,
                    cf_depths,
                    pix_coords,
                    lb_sched,
                    sorted_gs_id,
                )
                cf_render_alphas = self.tile2image(
                    cf_render_alphas.permute(1, 2, 0), height, width
                )
                cf_render_alphas = cf_render_alphas[0:1]  # Take any one channel
                # cf_render_alphas = torch.ones_like(cf_render_depths)

                render_colors.append(cf_render_colors.permute(1, 2, 0))
                render_depths.append(cf_render_depths.permute(1, 2, 0))
                render_alphas.append(cf_render_alphas.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)
        render_alphas = torch.stack(render_alphas)

        meta = {
            "gaussian_ids": gaussian_ids,
            "means2d": means2d,
            "radii": radius,
            "width": width,
            "height": height,
            "n_cameras": C,
        }
        return render_colors, render_depths, render_alphas, meta


def new_ascend_rasterization(
    ascend_render: Rasterizer,
    means: Tensor,  # [N, 3] or [T, N, 3]
    quats: Tensor,  # [N, 4] or [T, N, 4]
    scales: Tensor,  # [N, 3] or [T, N, 3]
    opacities: Tensor,  # [N] or [T, N]
    colors: Tensor,  # [N, D] or [T, N, D]
    viewmats: Tensor,  # [C, 4, 4] or [T, C, 4, 4]
    Ks: Tensor,  # [C, 3, 3] or [T, C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 32,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    batch_per_iter: int = 100,
    packed: bool = False,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict]:

    assert len(means.shape) == 3
    assert len(quats.shape) == len(scales.shape) == len(colors.shape) == 3
    assert len(opacities.shape) == 2
    assert len(viewmats.shape) == len(Ks.shape) == 4
    assert means.shape[0] == quats.shape[0] == scales.shape[0] == colors.shape[0] == opacities.shape[0] \
        == viewmats.shape[0] == Ks.shape[0]

    B = means.shape[0]
    render_colors = []
    render_alphas = []
    for b in range(B):
        means_single_sample = means[b]
        quats_single_sample = quats[b]
        scales_single_sample = scales[b]
        opacities_single_sample = opacities[b]
        colors_single_sample = colors[b]
        viewmats_single_sample = viewmats[b]
        Ks_single_sample = Ks[b]
        splats = {
            "means": means_single_sample,
            "quats": quats_single_sample,
            "scales": scales_single_sample,
            "opacities": opacities_single_sample,
            "colors": colors_single_sample,
        }

        # kwargs = {
        #     "near_plane": near_plane,
        #     "far_plane": far_plane,
        #     "eps2d": eps2d,
        #     "sh_degree": None,
        #     "render_mode": render_mode,
        # }

        V = viewmats_single_sample.shape[0]
        assert Ks_single_sample.shape[0] == V

        cols_single_sample, alphas_single_sample = [], []

        for v in range(V):
            cols_single_view, alphas_single_view, _ = ascend_render.ascend_rasterize_splats(
                viewmats=viewmats_single_sample[v:v + 1].contiguous(),
                Ks=Ks_single_sample[v:v + 1],
                size=(width, height),
                tile_size=32,
                active_sh_degree=None,
                splats=splats,
            )
            cols_single_sample.append(cols_single_view)
            alphas_single_sample.append(alphas_single_view)

        cols_single_sample = torch.cat(cols_single_sample)
        alphas_single_sample = torch.cat(alphas_single_sample)

        render_colors.append(cols_single_sample)
        render_alphas.append(alphas_single_sample)

    render_colors = torch.stack(render_colors)
    render_alphas = torch.stack(render_alphas)

    return render_colors, render_alphas, None

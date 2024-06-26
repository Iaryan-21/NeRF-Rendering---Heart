from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from typing_extensions import Literal



import torch 
import torch.nn as nn
from nerfstudio.models.base_model import ModelConfig, Model


import torch
import nerfacc
import trimesh
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
from rich.console import Console
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc

from .utils import FocalLoss, AverageMeter
from .field import LightningField
from .sampler import LightningNeRFSampler




@dataclass 
class EnhancedNeRFModelConfig:
    near_plane: float = 0.01
    far_plane: float = 6.0 
    vi_mlp_num_layers: int = 8
    vi_mlp_hidden_size: int = 256
    vd_mlp_num_layers: int = 4
    vd_mlp_hidden_size: int = 128
    appearance_embedding_dim: int = 32
    use_average_appearence_embedding: bool= True
    background_color: Literal["random", "black", "white"] = "black"
    alpha_thre: float = 0.01
    cone_angle: float = 0.004 
    point_cloud_path: Optional[str] = None
    frontal_axis: Literal["x", "y"] = "x"
    init_density_value: float = 10.0
    density_grid_base_res: int = 256
    density_log2_hashmap_size: int = 24
    color_grid_base_res: int = 128
    color_grid_max_res: int = 2048
    color_grid_fpl: int = 2
    color_log2_hashmap_size: int = 19
    color_grid_num_levels: int = 8
    bg_density_grid_res: int = 32
    bg_density_log2_hashmap_size: int = 18
    bg_color_grid_base_res: int = 32
    bg_color_grid_max_res: int = 128
    bg_color_log2_hashmap_size: int = 16
    occ_grid_base_res: int = 256
    occ_grid_num_levels: int = 2
    occ_grid_update_warmup_step: int = 2
    occ_num_samples_per_ray: int = 1000
    # occ_num_samples_per_ray: int = 8
    pdf_num_samples_per_ray: int = 8
    pdf_samples_warmup_step: int = 500
    pdf_samples_fixed_step: int = 2000
    pdf_samples_fixed_ratio: float = 0.5
    rgb_padding: Optional[float] = None
    loss_coefficients: Dict[str, float] = field(default_factory=lambda:{
        "rgb_loss": 1.0,
        "res_rgb_loss":0.01,
    })

class EnhancedNeRFModel(Model):
    config: EnhancedNeRFModelConfig

    def __init__(self, config: EnhancedNeRFModelConfig):
        super().__init__()
        self.config = config

        self.vi_mlp = nn.Sequential(
            nn.Linear(3, self.config.vi_mlp_hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.config.vi_mlp_hidden_size, self.config.vi_mlp_hidden_size), nn.ReLU()) for _ in range(self.config.vi_mlp_num_layers - 1)]

        )   

        self.vd_mlp = nn.Sequential(
            nn.Linear(self.config.vi_mlp_hidden_size + 3, self.config.vd_mlp_hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.config.vd_mlp_hidden_size, self.config.vd_mlp_hidden_size), nn.ReLU()) for _ in range (self.config.vd_mlp_num_layers - 1)],
            nn.Linear(self.config.vd_mlp_hidden_size, 4)
        )

        if self.config.appearance_embedding_dim > 0:
            self.appearance_embedding = nn.Embedding(self.config.num_images, self.config.appearance_embedding_dim)
    
    def forward(self, x):
        vi_features = self.vi_mlp(x[:,:3])
        if hasattr(self, 'appearance_embedding')  and x.shape[1] > 3:
            appearance_idx = x[:,3].long()
            vi_features = torch.cat([vi_features, self.appearance_embedding(appearance_idx)], dim=1)
        return self.vd_mlp(vi_features)

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            if self.training_step < self.config.pdf_samples_warmup_step:
                num_fine_samples = self.config.pdf_num_samples_per_ray
            elif self.training_step < self.config.pdf_samples_fixed_step:
                ss = self.config.pdf_num_samples_per_ray
                fixed_step = self.config.pdf_samples_fixed_step
                warmup_step = self.config.pdf_samples_fixed_step
                max_ratio = 1. -self.config.pdf_samples_fixed_ratio
                ratio = (self.training_step.item() - warmup_step) / (fixed_step - warmup_step)
                num_fine_samples = round(ss*(1-max_ratio*ratio))
            else:
                 num_fine_samples = round(self.config.pdf_num_samples_per_ray*self.config.pdf_samples_fixed_ratio)
            ray_samples, ray_indices = self.sampler(
                ray_bundle = ray_bundle,
                near_plane = self.config.near_plane,
                far_plane = self.config.far_plane,
                render_step_size = self.render_step_size,
                alpha_thre = self.config.alpha_thre,
                cone_angle = self.config.cone_angle,
                num_fine_samples = num_fine_samples,
            )

            field_outputs = self.field(ray_samples)

            sigmas = field_outputs[FieldHeadNames.DENSITY]
            colors = field_outputs[FieldHeadNames.RGB]

            packed_info = nerfacc.pack_info(ray_indices, num_rays)
            weights, _, _ = nerfacc.render_weight_from_density(
                t_starts = ray_samples.frustums.starts[..., 0],
                t_ends = ray_samples.frustums.ends[..., 0],
                sigmas = sigmas[..., 0],
                packed_info = packed_info,
            )

            if not self.training:
                 colors = torch.nan_to_num(colors)
        comp_rgb = nerfacc.accumulate_along_rays(
            weights, values=colors, ray_indices=ray_indices, n_rays=num_rays
        )
        accumulation = nerfacc.accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=num_rays
        )
        bg_color = torch.rand_like(comp_rgb) if self.bg_color is None else self.bg_color
        rgb = comp_rgb + bg_color * (1.0 - accumulation)
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        depth = nerfacc.accumulate_along_rays(
            weights, values=steps, ray_indices=ray_indices, n_rays=num_rays
        )
        depth = depth / (accumulation + 1e-10)
        depth = torch.clip(depth, steps.min(), steps.max())
        
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        
        if self.training:
            res_rgb = field_outputs.get("res_rgb", None)
            if res_rgb is not None:
                res_rgb_loss = torch.mean(res_rgb)  # for rgb combined before sigmoid
                outputs.update({"res_rgb_loss": res_rgb_loss})

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        loss_dict = {"rgb_loss": self.rgb_loss(image, outputs["rgb"])}
        # regularizer for view-dependent color
        if outputs.get("res_rgb_loss") is not None:
            loss_dict.update({"res_rgb_loss": outputs["res_rgb_loss"]})
        
        loss_dict = misc.scale_dict(loss_dict, se
                



    




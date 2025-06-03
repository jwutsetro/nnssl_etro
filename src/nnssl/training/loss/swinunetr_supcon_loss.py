from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialContrastLoss(nn.Module):
    """Contrastive loss using spatial proximity to define positives."""

    def __init__(self, batch_size: int, device: torch.device, temperature: float = 0.5, pos_threshold_mm: float = 30.0):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.pos_threshold = pos_threshold_mm
        self.register_buffer("eps", torch.tensor(1e-8))

    def forward(self, features: torch.Tensor, coords_mm: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features, dim=1)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        dist = torch.cdist(coords_mm, coords_mm)
        device = features.device
        eye = torch.eye(dist.shape[0], device=device, dtype=torch.bool)
        pos_mask = (dist < self.pos_threshold) & (~eye)

        exp_sim = torch.exp(sim / self.temp) * (~eye)
        denom = exp_sim.sum(dim=1)
        numer = (exp_sim * pos_mask.float()).sum(dim=1)
        loss = -torch.log((numer + self.eps) / (denom + self.eps))
        return loss.mean()


class SwinUNETRSupConLoss(nn.Module):
    def __init__(self, batch_size: int, device: torch.device, rec_loss_weight: float, contrast_loss_weight: float, rot_loss_weight: float, pos_threshold_mm: float = 30.0):
        super().__init__()
        self.rec_loss = nn.L1Loss().to(device)
        self.contrast_loss = SpatialContrastLoss(batch_size, device, pos_threshold_mm=pos_threshold_mm).to(device)
        self.rot_loss = nn.CrossEntropyLoss().to(device)

        self.rec_loss_weight = rec_loss_weight
        self.contrast_loss_weight = contrast_loss_weight
        self.rot_loss_weight = rot_loss_weight

    def __call__(self, rotations_pred: torch.Tensor, rotations: torch.Tensor, contrast: torch.Tensor, coords_mm: torch.Tensor, reconstructions: torch.Tensor, imgs_rotated: torch.Tensor) -> torch.Tensor:
        rec_loss = self.rec_loss(reconstructions, imgs_rotated)
        contrast_loss = self.contrast_loss(contrast, coords_mm)
        rot_loss = self.rot_loss(rotations_pred, rotations)
        return (
            self.rec_loss_weight * rec_loss
            + self.contrast_loss_weight * contrast_loss
            + self.rot_loss_weight * rot_loss
        )

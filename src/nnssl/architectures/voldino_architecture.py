import torch
from torch import nn
from nnssl.architectures.voco_architecture import VocoProjectionHead

class VolDINOArchitecture(nn.Module):
    """Simple DINO style architecture providing global and patch embeddings."""

    def __init__(self, encoder: nn.Module, feature_channels: list[int]):
        super().__init__()
        self.encoder = encoder
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        total = sum(feature_channels)
        self.global_head = VocoProjectionHead(total, 1024, 1024, norm_op=nn.InstanceNorm1d)
        self.patch_head = nn.Conv3d(feature_channels[0], 256, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        if isinstance(feats, list):
            global_feat = torch.cat([self.global_pool(f) for f in feats], dim=1)
            patch_feat = feats[0]
        else:
            global_feat = self.global_pool(feats)
            patch_feat = feats
        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_out = self.global_head(global_feat)
        patch_out = self.patch_head(patch_feat)
        return global_out, patch_out

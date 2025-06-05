from typing import Tuple, Literal
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import Rot90Transform, MirrorTransform, SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.abstract_transforms import Compose
from einops import rearrange
from loguru import logger

class VolDINOTransform(AbstractTransform):
    """Create global and local 3D crops similar to DINO multi-crop."""

    def __init__(
        self,
        global_crop_size: Tuple[int, int, int],
        local_crop_size: Tuple[int, int, int],
        n_global: int = 2,
        n_local: int = 4,
        aug: Literal["train", "none"] = "train",
        data_key: str = "data",
    ):
        self.data_key = data_key
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.n_global = n_global
        self.n_local = n_local
        self.aug = aug

        if aug == "train":
            self.crop_augmentations: Compose = Compose(
                [
                    GaussianNoiseTransform(p_per_sample=0.1),
                    GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
                    BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
                    ContrastAugmentationTransform(p_per_sample=0.15),
                    SimulateLowResolutionTransform(
                        zoom_range=(0.5, 1),
                        per_channel=True,
                        p_per_channel=0.5,
                        order_downsample=0,
                        order_upsample=3,
                        p_per_sample=0.1,
                        ignore_axes=None,
                    ),
                    GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
                    GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3),
                    MirrorTransform(axes=(0,)),
                    MirrorTransform(axes=(1,)),
                    MirrorTransform(axes=(2,)),
                    Rot90Transform(axes=(0, 1)),
                ]
            )

    def random_crop(self, data: np.ndarray, crop_size: Tuple[int, int, int]) -> np.ndarray:
        """Randomly crop ``data`` which is shaped [C, X, Y, Z]."""
        x_off = np.random.randint(0, data.shape[1] - crop_size[0] + 1)
        y_off = np.random.randint(0, data.shape[2] - crop_size[1] + 1)
        z_off = np.random.randint(0, data.shape[3] - crop_size[2] + 1)
        return data[
            :,
            x_off : x_off + crop_size[0],
            y_off : y_off + crop_size[1],
            z_off : z_off + crop_size[2],
        ]

    def get_crops(self, data: np.ndarray, crop_size: Tuple[int, int, int], n_crops: int) -> np.ndarray:
        crops = [self.random_crop(data, crop_size) for _ in range(n_crops)]
        crops = np.stack(crops, axis=0)
        return crops

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        B = data.shape[0]
        global_crops = []
        local_crops = []
        for b in range(B):
            g = self.get_crops(data[b], self.global_crop_size, self.n_global)
            l = self.get_crops(data[b], self.local_crop_size, self.n_local)
            global_crops.append(g)
            local_crops.append(l)
        global_crops = np.concatenate(global_crops, axis=0)
        local_crops = np.concatenate(local_crops, axis=0)

        if self.aug == "train":
            global_crops = self.crop_augmentations(**{"data": global_crops, "seg": None})["data"]
            local_crops = self.crop_augmentations(**{"data": local_crops, "seg": None})["data"]

        batch = {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "batch_size": B,
        }
        return batch


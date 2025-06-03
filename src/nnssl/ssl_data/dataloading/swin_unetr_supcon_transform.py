from __future__ import annotations

from .swin_unetr_transform import SwinUNETRTransform


class SwinUNETRSupConTransform(SwinUNETRTransform):
    """Transform that keeps patch centroids for supervised contrastive learning."""

    def __init__(self, data_key: str = "data", centroid_key: str = "centroids"):
        super().__init__(data_key)
        self.centroid_key = centroid_key

    def __call__(self, **data_dict):
        centroids = data_dict.get(self.centroid_key)
        res = super().__call__(**data_dict)
        if centroids is not None:
            res[self.centroid_key] = centroids
        return res

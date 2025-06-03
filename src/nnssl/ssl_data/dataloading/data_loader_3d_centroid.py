from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple

from .data_loader_3d import nnsslDataLoader3D
from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2


class nnsslDataLoader3DCentroid(nnsslDataLoader3D):
    """Extension of :class:`nnsslDataLoader3D` that also returns patch centroids."""

    def __init__(
        self,
        data: nnSSLDatasetBlosc2,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
    ) -> None:
        super().__init__(data, batch_size, patch_size, final_patch_size, sampling_probabilities, pad_sides)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        anon_all = np.zeros(self.data_shape, dtype=np.uint8)
        centroids = np.zeros((len(selected_keys), 3), dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            data, anon, anat, properties = self._data[i]
            if anon is None:
                anon = np.zeros(data.shape, dtype=np.uint8)
            case_properties.append(properties)

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape)

            valid_bbox_lbs = [max(0, bbox_lbs[d]) for d in range(dim)]
            valid_bbox_ubs = [min(shape[d], bbox_ubs[d]) for d in range(dim)]

            this_slice = (
                [slice(0, data.shape[0])] + [slice(l, u) for l, u in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[tuple(this_slice)]
            anon = anon[tuple(this_slice)]

            padding = [(-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0)) for d in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), "constant", constant_values=0)
            anon_all[j] = np.pad(anon, ((0, 0), *padding), "constant", constant_values=0)

            spacing = np.array(properties.get("spacing", (1.0, 1.0, 1.0)), dtype=float)
            center_voxel = np.array(bbox_lbs) + np.array(self.patch_size) / 2.0
            centroids[j] = center_voxel * spacing

        return {
            "data": data_all,
            "seg": anon_all,
            "centroids": centroids,
            "properties": case_properties,
            "keys": selected_keys,
        }

from __future__ import annotations

import torch
from typing_extensions import override

from nnssl.ssl_data.dataloading.swin_unetr_supcon_transform import SwinUNETRSupConTransform
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from nnssl.ssl_data.dataloading.data_loader_3d_centroid import nnsslDataLoader3DCentroid
from nnssl.training.loss.swinunetr_supcon_loss import SwinUNETRSupConLoss
from nnssl.training.nnsslTrainer.swinunetr_pretrain.SwinUNETRTrainer import SwinUNETRTrainer
from nnssl.experiment_planning.experiment_planners.plan import Plan
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class SwinUNETRSupConTrainer(SwinUNETRTrainer):
    """Swin UNETR trainer using supervised contrastive learning based on spatial proximity."""

    @override
    def build_loss(self):
        return SwinUNETRSupConLoss(
            self.batch_size,
            self.device,
            self.rec_loss_weight,
            self.contrast_loss_weight,
            self.rot_loss_weight,
        )

    @override
    def get_plain_dataloaders(self, initial_patch_size):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnsslDataLoader3DCentroid(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslDataLoader3DCentroid(
            dataset_val,
            self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = [
            SwinUNETRSupConTransform(),
            NumpyToTensor(cast_to="float", keys=["imgs_rotated", "imgs_rotated_cutout", "centroids"]),
            NumpyToTensor(cast_to="long", keys="rotations"),
        ]
        return Compose(tr_transforms)

    @override
    def get_dataloaders(self):
        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=self.config_plan.patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    @override
    def train_step(self, batch: dict) -> dict:
        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]
        centroids = batch["centroids"]

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)
        centroids = centroids.repeat(2, 1)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)
        centroids = centroids.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
            l = self.loss(rotations_pred, rotations, contrast_pred, centroids, reconstructions, imgs_rotated)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}


class SwinUNETRSupConTrainer_BS2(SwinUNETRSupConTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2


class SwinUNETRSupConTrainer_BS8(SwinUNETRSupConTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8

    @override
    def validation_step(self, batch: dict) -> dict:
        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]
        centroids = batch["centroids"]

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)
        centroids = centroids.repeat(2, 1)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)
        centroids = centroids.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
                l = self.loss(rotations_pred, rotations, contrast_pred, centroids, reconstructions, imgs_rotated)

        return {"loss": l.detach().cpu().numpy()}

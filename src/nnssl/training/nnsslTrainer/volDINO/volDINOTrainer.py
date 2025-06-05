from copy import deepcopy
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from typing_extensions import override
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch.cuda.amp import autocast

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.voldino_architecture import VolDINOArchitecture
from nnssl.training.loss.voldino_loss import VolDINOLoss
from nnssl.ssl_data.dataloading.voldino_transform import VolDINOTransform
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.utilities.helpers import dummy_context
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnssl.experiment_planning.experiment_planners.plan import Plan, ConfigurationPlan
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    create_blocky_mask,
)

class VolDINOTrainer(AbstractBaseTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple[int, int, int] = (160, 160, 160),
    ):
        plan.configurations[configuration_name].patch_size = patch_size
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.teacher_momentum = 0.996
        self.global_crop_size = tuple(self.config_plan.patch_size)
        self.local_crop_size = tuple(s // 2 for s in self.global_crop_size)
        self.n_global = 2
        self.n_local = 4
        self.mask_ratio = 0.5

    def initialize(self):
        super().initialize()
        self.teacher = deepcopy(self.network)
        for p in self.teacher.parameters():
            p.requires_grad = False

    def build_loss(self) -> nn.Module:
        return VolDINOLoss(out_dim=1024)

    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> tuple[nn.Module, AdaptationPlan]:
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        architecture = VolDINOArchitecture(encoder, encoder.output_channels)

        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.global_crop_size

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=num_input_channels,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        return architecture, adapt_plan

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        tr_transforms = [
            VolDINOTransform(
                global_crop_size=patch_size,
                local_crop_size=tuple([s // 2 for s in patch_size]),
                n_global=2,
                n_local=4,
                aug="train",
                data_key="data",
            ),
            NumpyToTensor(["global_crops", "local_crops"], "float"),
        ]
        return Compose(tr_transforms)

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        val_transforms = [
            VolDINOTransform(global_crop_size=(160,160,160), local_crop_size=(80,80,80), aug="none"),
            NumpyToTensor(["global_crops", "local_crops"], "float"),
        ]
        return Compose(val_transforms)

    def get_dataloaders(self):
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

        tr_transforms = self.get_training_transforms(patch_size, rotation_for_DA, mirror_axes, do_dummy_2d_data_aug)
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(patch_size)

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

    @staticmethod
    def mask_creation(
        batch_size: int,
        patch_size: Tuple[int, int, int],
        mask_ratio: float,
        rng_seed: int | None = None,
        block_size: int = 16,
    ) -> torch.Tensor:
        mask = [
            create_blocky_mask(patch_size, block_size, mask_ratio, rng_seed)
            for _ in range(batch_size)
        ]
        mask = torch.stack(mask)[:, None, ...]
        return mask

    def update_teacher(self):
        for p_s, p_t in zip(self.network.parameters(), self.teacher.parameters()):
            p_t.data.mul_(self.teacher_momentum).add_(p_s.data * (1.0 - self.teacher_momentum))

    def train_step(self, batch: dict) -> dict:
        g_crops = batch["global_crops"].to(self.device, non_blocking=True)
        l_crops = batch["local_crops"].to(self.device, non_blocking=True)
        if l_crops.shape[2:] != self.global_crop_size:
            l_crops = torch.nn.functional.interpolate(
                l_crops,
                size=self.global_crop_size,
                mode="trilinear",
                align_corners=False,
            )
        B = batch["batch_size"]

        all_crops = torch.cat([g_crops, l_crops], 0)
        mask = self.mask_creation(all_crops.shape[0], all_crops.shape[2:], self.mask_ratio).to(self.device, non_blocking=True)
        masked_crops = all_crops * mask

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            student_global, student_patch = self.network(masked_crops)
            with torch.no_grad():
                teacher_global, teacher_patch = self.teacher(all_crops)
            img_loss = self.loss(student_global, teacher_global[: self.n_global * B])
            patch_loss = nn.functional.mse_loss(student_patch, teacher_patch)
            loss = img_loss + patch_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.optimizer.step()

        self.update_teacher()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        g_crops = batch["global_crops"].to(self.device, non_blocking=True)
        l_crops = batch["local_crops"].to(self.device, non_blocking=True)
        if l_crops.shape[2:] != self.global_crop_size:
            l_crops = torch.nn.functional.interpolate(
                l_crops,
                size=self.global_crop_size,
                mode="trilinear",
                align_corners=False,
            )
        all_crops = torch.cat([g_crops, l_crops], 0)
        mask = self.mask_creation(all_crops.shape[0], all_crops.shape[2:], self.mask_ratio).to(self.device, non_blocking=True)
        masked_crops = all_crops * mask
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                student_global, student_patch = self.network(masked_crops)
                teacher_global, teacher_patch = self.teacher(all_crops)
                img_loss = self.loss(student_global, teacher_global[: self.n_global * batch["batch_size"]])
                patch_loss = nn.functional.mse_loss(student_patch, teacher_patch)
                loss = img_loss + patch_loss
        return {"loss": loss.detach().cpu().numpy()}


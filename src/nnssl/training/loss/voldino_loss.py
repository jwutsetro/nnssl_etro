import torch
from torch import nn
import torch.nn.functional as F

class VolDINOLoss(nn.Module):
    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        # teacher centering and sharpening
        t_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        s_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        loss = -torch.sum(t_out * s_out, dim=-1).mean()
        # update center
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * teacher_output.mean(dim=0, keepdim=True)
        return loss

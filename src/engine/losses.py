import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _F_loss(nn.Module):
    """The F loss (regression)"""
    def __init__(self):
        super().__init__()

    def forward(self, regressor_logits, regressor_targets):
        mse_output = F.mse_loss(regressor_logits, regressor_targets)
        tanH_output = torch.tanh(mse_output)
        output = tanH_output
        return output


class _F_bar_loss(nn.Module):
    """The F̄ loss (regression). Its the mathematical complementar for F"""
    def __init__(self):
        super().__init__()

    def forward(self, classifier_logits, regressor_targets):
        mse_output = F.mse_loss(classifier_logits, regressor_targets)
        tanH_output = torch.tanh(mse_output)
        output = 1 - tanH_output
        return output

#TODO corrigir G e G_bar
class _G_loss(nn.Module):
    """The G loss (classification)"""
    def __init__(self):
        super().__init__()

    def forward(self, classifier_logits: torch.Tensor, classifier_targets: torch.Tensor) -> torch.Tensor:
        output = F.cross_entropy(classifier_logits, classifier_targets)
        return output


class _G_bar_loss(nn.Module):
    """The Ḡ loss (classification). Its the mathematical complementar for G"""
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, regressor_logits, classifier_targets):
        cross_entropy_output = F.cross_entropy(regressor_logits, classifier_targets, reduction="none")
        output = torch.clamp(1.0 - cross_entropy_output / self.tau, min=0.0).mean()
        return output

class LSFLoss(nn.Module):
    """Compound loss designed to force the latent space separation"""

    def __init__(self, weight_f: Optional[int]=1, weight_f_bar: Optional[int]=1, weight_g: Optional[int]=1, weight_g_bar: Optional[int]=1, tau_g: Optional[int]=1):
        super().__init__()
        self.f_loss = _F_loss()
        self.f_bar_loss = _F_bar_loss()
        self.g_loss = _G_loss()
        self.g_bar_loss = _G_bar_loss(tau=tau_g)
        self.weight_f = weight_f
        self.weight_f_bar = weight_f_bar
        self.weight_g = weight_g
        self.weight_g_bar = weight_g_bar

    def forward(self, regressor_logits, regressor_targets, classifier_logits, classifier_targets):
        loss_f = self.f_loss(regressor_logits, regressor_targets)
        loss_f_bar = self.f_bar_loss(classifier_logits, regressor_targets)
        loss_g = self.g_loss(classifier_logits, classifier_targets)
        loss_g_bar = self.g_bar_loss(regressor_logits, classifier_targets)

        total = (self.w_f * loss_f + self.w_f_bar * loss_f_bar + self.w_g * loss_g + self.w_g_bar * loss_g_bar)
        return {
                "total": total,
                "F": loss_f,
                "F_bar": loss_f_bar,
                "G": loss_g,
                "G_bar": loss_g_bar,
               }
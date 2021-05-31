import torch
import torch.nn.functional as F
from torch import nn


class DualDecompLoss(nn.Module):
    def __init__(self, attn_kl=True, kl_weight=1.0, detach_targets=True):
        super(DualDecompLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.attn_kl = attn_kl
        self.kl_weight = kl_weight
        self.detach_targets = detach_targets

    def forward(self, output, targets, marginals):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl:
            kl_losses = []
            for marg1, marg2 in marginals:
                if self.detach_targets:
                    marg2 = marg2.detach()
                kl_losses.append(F.kl_div(marg1.log(), marg2, reduction='batchmean'))
            total_loss = cross_entropy_loss + self.kl_weight * sum(kl_losses)
            return cross_entropy_loss, kl_losses, total_loss
        return (cross_entropy_loss, )
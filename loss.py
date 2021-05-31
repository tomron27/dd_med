import torch
import torch.nn.functional as F
from torch import nn


class DualDecompLoss(nn.Module):
    def __init__(self, attn_kl=True, alpha=1.0, detach_targets=True):
        super(DualDecompLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.attn_kl = attn_kl
        self.alpha = alpha
        self.detach_targets = detach_targets

    def forward(self, output, targets, marginals):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl:
            kl_losses = []
            marg_k, marg_l = marginals
            if self.detach_targets:
                marg_l = marg_l.detach()
            kl_losses.append(F.kl_div(marg_k.log(), marg_l, reduction='batchmean'))
            total_loss = cross_entropy_loss + self.alpha * sum(kl_losses)
            return cross_entropy_loss, kl_losses, total_loss
        return (cross_entropy_loss, )
import torch
import torch.nn.functional as F
from torch import nn


class SumPool(nn.Module):
    def __init__(self, factor=2):
        super(SumPool, self).__init__()
        self.factor = factor
        self.avgpool = nn.AvgPool2d(kernel_size=(factor, factor), stride=(factor, factor), padding=(0, 0))

    def forward(self, x):
        return self.avgpool(x) * (self.factor ** 2)


class SimpleSelfAttention(nn.Module):
    def __init__(self, input_channels, embed_channels, kernel_size=(1, 1), stride=(1, 1),
                 padding=(0, 0), name='simple_self_attn'):
        super(SimpleSelfAttention, self).__init__()
        self.w1 = nn.Conv2d(in_channels=input_channels,
                            out_channels=embed_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,)
        self.w2 = nn.Conv2d(in_channels=embed_channels,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.name = name

    def forward(self, x):
        tau = self.w1(x)
        tau = F.normalize(tau)
        tau = self.relu(tau)
        tau = self.w2(tau).squeeze(1)
        tau = torch.softmax(tau.flatten(1), dim=1).reshape(tau.shape)
        attended_x = torch.einsum('bcxy,bxy->bcxy', x, tau)
        return attended_x, tau


class Marginals(nn.Module):
    def __init__(self, margin_dim=256, factor=2):
        super(Marginals, self).__init__()
        self.factor = factor
        self.margin_dim = margin_dim
        self.tau_pool = SumPool(factor=factor)
        self.lamb = nn.Parameter(torch.ones(1, 1, margin_dim // 8, margin_dim // 8))
        self.name = "marginals_extended"

    def forward(self, tau_k, tau_l):
        tau_k_lamb_neg = SumPool(2)(tau_k) * torch.exp(-self.lamb)
        tau_k_lamb_neg = tau_k_lamb_neg / tau_k_lamb_neg.view(tau_k_lamb_neg.shape[0], -1).sum(dim=1)

        tau_l_lamb = tau_l * torch.exp(self.lamb)
        tau_l_lamb = tau_l_lamb / tau_l_lamb.view(tau_l_lamb.shape[0], -1).sum(dim=1)
        return tau_k_lamb_neg, tau_l_lamb

"""EEGNet (Lawhern et al., 2018), implemented in PyTorch following the paper's architecture.

Reference: V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces", J. Neural Eng. 15 (2018) 056013.

Block 1:  temporal conv (F1 filters, length K)          -> BatchNorm
          depthwise spatial conv (D per filter, max-norm 1) -> BatchNorm -> ELU -> AvgPool(4) -> Dropout
Block 2:  separable conv = depthwise temporal conv (length K2) + pointwise 1x1 conv (F2)
          -> BatchNorm -> ELU -> AvgPool(8) -> Dropout
Head:     flatten -> dense layer (max-norm 0.25) -> class logits
"""
import torch
import torch.nn.functional as Fn
from torch import nn


class FTDropBlock(nn.Module):
    """FT-DropBlock (Sedi Nzakuna et al., I2MTC 2025): drops contiguous blocks of
    (feature maps x time points) instead of single activations; PyTorch port of their Keras layer.

    Input (B, F, 1, T) as in EEGNet after the spatial convolution. A block is dropped where a
    seed (probability gamma = p / block_size^2) lies within block_size // 2 in both axes;
    the output is rescaled so its expected sum is unchanged. Inactive in eval mode.
    """

    def __init__(self, drop_prob=0.5, block_size=15):
        super().__init__()
        self.p, self.bs = drop_prob, block_size

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, F, _, T = x.shape
        gamma = self.p / self.bs ** 2
        seeds = (torch.rand(B, 1, F, T, device=x.device) < gamma).float()
        pad = self.bs // 2
        block = Fn.max_pool2d(Fn.pad(seeds, (pad, pad, pad, pad)), self.bs, stride=1)[:, :, :F, :T]
        keep = (1.0 - block).squeeze(1).unsqueeze(2)                 # (B, F, 1, T)
        return x * keep * (keep.numel() / keep.sum().clamp_min(1.0))


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_times=1000, n_classes=4, F1=8, D=2, F2=16,
                 kernel_length=125, separable_kernel_length=31, pool1=4, pool2=8,
                 dropout=0.5, bn_momentum=0.01, bn_eps=1e-3,
                 spatial_max_norm=1.0, dense_max_norm=0.25, drop_type="dropout", block_size=15):
        super().__init__()
        drop = (lambda: FTDropBlock(dropout, block_size)) if drop_type == "ftdb" else (lambda: nn.Dropout(dropout))
        self.spatial_max_norm = spatial_max_norm
        self.dense_max_norm = dense_max_norm

        # Block 1: temporal filters (learned band-pass filters), then spatial filters per temporal filter.
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=bn_momentum, eps=bn_eps)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=bn_momentum, eps=bn_eps)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = drop()

        # Block 2: separable convolution = depthwise (per feature map) + pointwise (mix feature maps).
        self.conv_separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, separable_kernel_length),
                                              padding="same", groups=F1 * D, bias=False)
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=bn_momentum, eps=bn_eps)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = drop()

        self.elu = nn.ELU()
        n_features = F2 * ((n_times // pool1) // pool2)
        self.classifier = nn.Linear(n_features, n_classes)
        self.apply_constraints()

    def forward(self, x):
        """x: (batch, channels, time) -> logits: (batch, n_classes). Softmax is applied outside."""
        x = x.unsqueeze(1)                                   # (B, 1, C, T)
        x = self.bn1(self.conv_temporal(x))                  # (B, F1, C, T)
        x = self.elu(self.bn2(self.conv_spatial(x)))         # (B, F1*D, 1, T)
        x = self.drop1(self.pool1(x))                        # (B, F1*D, 1, T/4)
        x = self.conv_separable_point(self.conv_separable_depth(x))
        x = self.elu(self.bn3(x))                            # (B, F2, 1, T/4)
        x = self.drop2(self.pool2(x))                        # (B, F2, 1, T/32)
        return self.classifier(x.flatten(start_dim=1))

    @torch.no_grad()
    def apply_constraints(self):
        """Max-norm constraints from the paper; call after every optimizer step.

        Each output filter's weight vector is rescaled to have L2 norm <= max_norm
        (same as Keras max_norm(axis=0) used in the original implementation).
        """
        self.conv_spatial.weight.data = torch.renorm(self.conv_spatial.weight.data, p=2, dim=0,
                                                     maxnorm=self.spatial_max_norm)
        self.classifier.weight.data = torch.renorm(self.classifier.weight.data, p=2, dim=0,
                                                   maxnorm=self.dense_max_norm)

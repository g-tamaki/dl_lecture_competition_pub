import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        consider_subjects: bool = True,
    ) -> None:
        super().__init__()

        self.consider_subjects = consider_subjects

        self.blocks1 = nn.Conv1d(in_channels, in_channels, 1, padding="same")

        if self.consider_subjects:
            self.hid_dim = hid_dim
            self.num_subjects = 4

            self.blocks = nn.ModuleList([nn.Sequential(
                # ConvBlock(in_channels, hid_dim, dilation1=1, dilation2=1, kernel_size=1),
                # nn.Linear(in_channels, hid_dim),
                nn.Conv1d(in_channels, hid_dim, 1, padding="same"),
                # ConvBlock(hid_dim, hid_dim, dilation1=128, dilation2=256),
            ) for i in range(self.num_subjects)])

        else:
            self.blocks = nn.Sequential(
                ConvBlock(in_channels, hid_dim, dilation1=1, dilation2=4),
                ConvBlock(hid_dim, hid_dim, dilation1=8, dilation2=16),
                ConvBlock(hid_dim, hid_dim, dilation1=32, dilation2=64),
                # ConvBlock(hid_dim, hid_dim, dilation1=128, dilation2=256),
            )

        self.head = nn.Sequential(
            ConvBlock(hid_dim, hid_dim, dilation1=4, dilation2=8),
            ConvBlock(hid_dim, hid_dim, dilation1=16, dilation2=32),
            ConvBlock(hid_dim, hid_dim, dilation1=64, dilation2=128),
            # ConvBlock(hid_dim, hid_dim, dilation1=1, dilation2=4),
            # ConvBlock(hid_dim, hid_dim, dilation1=8, dilation2=16),
            # ConvBlock(hid_dim, hid_dim, dilation1=32, dilation2=64),
            # ConvBlock(hid_dim, hid_dim, dilation1=128, dilation2=256),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.array([1], dtype='float32')))

    def forward(self, X: torch.Tensor, subject_ids) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # X = self.blocks(X)
        
        batch_size = X.size(0)

        X = self.blocks1(X)

        if self.consider_subjects:
            x1 = torch.zeros(batch_size, self.hid_dim, 281).to(X.device)
            for i in range(self.num_subjects):
                x1[subject_ids==i] = self.blocks[i](X[subject_ids==i])
            # for i, x in enumerate([self.blocks[i](X[subject_ids==i]) for i in range(self.num_subjects)]):
            #     x1[subject_ids==i] = x  # 早くならなかった
        else:
            x1 = self.blocks(X)

        return self.head(x1), self.logit_scale


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
        dilation1 = 1,
        dilation2 = 1,
        # dilation3 = 2,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same", dilation=dilation1)
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same", dilation=dilation2)
        # self.conv2 = nn.Conv1d(out_dim, 2*out_dim, kernel_size, padding="same", dilation=dilation3)  # paddingなくす？
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
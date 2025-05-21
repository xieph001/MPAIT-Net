"""
A Fault Diagnosis Method for Critical Rotating Components in Trains Based on Multi-Pooling Attention Convolution and an
Improved Vision Transformer
********************************************************************
*                                                                  *
* Copyright © 2025 All rights reserved                             *
* Written by Mr.XiePenghui                                         *
* [January 18,2025]                                                *
*                                                                  *
********************************************************************
"""
import torch
from torch import nn
from einops import rearrange
from utils import PolaLinearAttention, SpatialAttention


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        return self.activation(out)


class MPAConvLayer(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor):
        super(MPAConvLayer, self).__init__()
        self.groups = groups

        # Depthwise convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()

        # Pointwise convolution with grouping
        self.conv2 = nn.Conv2d(in_channels, hidden, kernel_size=1, groups=groups)
        self.batch_norm2 = nn.BatchNorm2d(hidden)

        self.channel_shuffle = self._channel_shuffle()

        # Restore original channel count
        self.conv3 = nn.Conv2d(hidden, in_channels, kernel_size=1, groups=groups)
        self.batch_norm3 = nn.BatchNorm2d(in_channels)

        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(reduction_factor)

    def _channel_shuffle(self):
        def shuffle(x):
            batch_size, num_channels, height, width = x.size()
            channels_per_group = num_channels // self.groups

            # Reshape and shuffle channels
            x = x.view(batch_size, self.groups, channels_per_group, height, width)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(batch_size, -1, height, width)
            return x

        return shuffle

    def forward(self, x):
        residual = x

        # Depthwise convolution
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.activation(out)

        # Pointwise convolution
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.activation(out)

        # Shuffle channels to mix information across groups
        out = self.channel_shuffle(out)

        # Restore original channel count
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.activation(out)

        # Apply spatial attention
        att = self.spatial_attention(out)
        out = out * att

        # Add residual connection
        return out + residual


class MPAConv(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            MPAConvLayer(
                in_channels=in_channels,
                hidden=hidden,
                groups=groups,
                reduction_factor=reduction_factor
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out = self.nn1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.nn2(out)
        return self.dropout(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_patches, sr_ratio=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    PolaLinearAttention(
                        dim=dim,
                        num_patches=num_patches,
                        num_heads=heads,
                        sr_ratio=sr_ratio,
                        kernel_size=5,
                        alpha=4
                    )
                ]),
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    MLP_Block(dim, mlp_dim, dropout=dropout)
                ])
            ]))

    def forward(self, x, H, W):
        for attention_block, mlp_block in self.layers:
            # Attention block with residual connection
            norm_x = attention_block[0](x)
            attention_out = attention_block[1](norm_x, H, W)
            x = attention_out + x

            # MLP block with residual connection
            norm_x = mlp_block[0](x)
            mlp_out = mlp_block[1](norm_x)
            x = mlp_out + x
        return x


class MPAIT_Net(nn.Module):
    def __init__(self, *, num_classes, num_patches, dim, depth, heads, mlp_dim, linear_dim, dropout=0.1, sr_ratio=1):
        super(MPAIT_Net, self).__init__()

        self.downsample_1 = DownsampleLayer(in_channels=2, out_channels=12, kernel_size=2, stride=2)
        self.mpaconv_1 = MPAConv(in_channels=12, hidden=48, groups=4, reduction_factor=16, depth=1)
        self.downsample_2 = DownsampleLayer(in_channels=12, out_channels=24, kernel_size=2, stride=2)
        self.mpaconv_2 = MPAConv(in_channels=24, hidden=96, groups=4, reduction_factor=8, depth=1)
        self.downsample_3 = DownsampleLayer(in_channels=24, out_channels=48, kernel_size=2, stride=2)
        self.mpaconv_3 = MPAConv(in_channels=48, hidden=192, groups=4, reduction_factor=4, depth=1)
        self.downsample_4 = DownsampleLayer(in_channels=48, out_channels=96, kernel_size=2, stride=2)
        self.mpaconv_4 = MPAConv(in_channels=96, hidden=384, groups=4, reduction_factor=2, depth=1)
        self.downsample_5 = DownsampleLayer(in_channels=96, out_channels=192, kernel_size=2, stride=2)

        self.patch_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_patches=num_patches,
            sr_ratio=sr_ratio
        )

        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, linear_dim)
        self.nn2 = nn.Linear(linear_dim, num_classes)

    def get_transformer_features(self, x, mask=None):
        out = self.downsample_1(x)
        out = self.mpaconv_1(out)
        out = self.downsample_2(out)
        out = self.mpaconv_2(out)
        out = self.downsample_3(out)
        out = self.mpaconv_3(out)
        out = self.downsample_4(out)
        out = self.mpaconv_4(out)
        out = self.downsample_5(out)
        out = self.patch_conv(out)
        B, C, H, W = out.shape
        out = rearrange(out, 'b c h w -> b (h w) c')
        out = self.transformer(out, H, W)
        out = torch.mean(out, dim=1)
        return out

    def forward(self, x, mask=None):
        out = self.get_transformer_features(x, mask)
        out = self.nn1(out)
        output = self.nn2(out)
        return output

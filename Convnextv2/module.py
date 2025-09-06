import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math
from Convnextv2.hat_arch import CAB

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectify(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectify, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.CAB = CAB(num_feat=dim, compress_ratio=1, squeeze_factor=1)
        self.CAB2 = CAB(num_feat=dim, compress_ratio=1, squeeze_factor=1)
        self.channel_emb = ChannelEmbed1(in_channels=dim, out_channels=dim, reduction=1,
                                         norm_layer=nn.BatchNorm2d)

        self.norm = nn.BatchNorm2d(dim)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        c1 = self.CAB(x1)
        c2 = self.CAB2(x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * c2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * c1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2




class ChannelEmbed1(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed1, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):

        x = self.channel_embed(x)
        return x

class fuse6(nn.Module):
    def __init__(self, dim):
        super(fuse6, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.CBR = nn.Sequential(
            nn.Conv2d(2 * dim, 2 *dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(2*dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()  # softmax和tanh区间相同，都是-1到1，sigmoid是0到1
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4, self.dim),
            nn.Sigmoid())

        self.channel_emb = ChannelEmbed1(in_channels=dim, out_channels=dim, reduction=1,
                                        norm_layer=nn.BatchNorm2d)

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x11 = self.sigmoid(x1)
        x22 = self.sigmoid(x2)

        x1_m = self.max_pool(x11)
        x2_m = self.max_pool(x22)

        x1_a = self.avg_pool(x11)
        x2_a = self.avg_pool(x22)

        y1 = x1_m * x2_m
        y2  = x1_a * x2_a

        concat = torch.cat((y1,y2),dim=1)
        sig = self.conv2(concat)

        y11 = x1 * sig + x1
        y22 = x2 * (1-sig) + x2

        x111 = self.avg_pool(y11)
        x222 = self.avg_pool(y22)
        x111 = self.sigmoid(x111)
        x222 = self.sigmoid(x222)

        y3 = y11 * x111
        y4 = y22 * x222

        y= y3+ y4

        y = self.channel_emb(y, H, W)

        return y



import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# FFL
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Local feature
class Local(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = int(dim // growth_rate)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.weight(y)
        return x*y


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        self.local = Local(dim, ffn_scale)
        self.gobal = Gobal(dim)
        self.conv = nn.Conv2d(2*dim, dim, 1, 1, 0)
        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_l = self.local(y)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y_l, y_g], dim=1)) + x

        y = self.fc(self.norm2(y)) + y
        return y
    

class ResBlock(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, b=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return res + x


class MixNet(nn.Module):
    def __init__(self, dim=48, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
            nn.PixelUnshuffle(upscaling_factor)
        )
        out_dim = upscaling_factor * dim 
        self.feats = nn.Sequential(*[AttBlock(out_dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(out_dim, dim, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x
    
# for LOL dataset
# class MixNet(nn.Module):
#     def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
#         super().__init__()
#         self.to_feat = nn.Sequential(
#             nn.Conv2d(3, dim, 3, 1, 1),
#             ResBlock(dim, 3, 1, 1)
#         )

#         self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

#         self.to_img = nn.Sequential(
#             ResBlock(dim, 3, 1, 1),
#             nn.Conv2d(dim, 3, 3, 1, 1)
#         )

#     def forward(self, x):
#         x = self.to_feat(x)
#         x = self.feats(x) + x
#         x = self.to_img(x)
#         return x

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

# LayerNorm的实现
# class LayerNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-12, device=None):
#         super(LayerNorm, self).__init__()
#
#         # 初始化 gamma 和 beta 为可训练参数
#         self.gamma = nn.Parameter(torch.ones(d_model, device=device))  # 初始化为全为1的向量
#         self.beta = nn.Parameter(torch.zeros(d_model, device=device))  # 初始化为全为0的向量
#
#         self.eps = eps
#         self.device = device
#
#     def forward(self, x):
#         # 将输入移动到与 gamma 和 beta 相同的设备
#         x = x.to(self.device)
#
#         mean = x.mean(dim=-1, keepdim=True)  # 计算均值
#         var = x.var(dim=-1, unbiased=False, keepdim=True)  # 计算方差
#
#         # 标准化输入
#         x_hat = (x - mean) / torch.sqrt(var + self.eps)
#
#         # 应用缩放和偏移
#         y = self.gamma * x_hat + self.beta
#         return y

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super(VAE_AttentionBlock, self).__init()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [Batch_size, channels, Height, Width]

        residual = x

        n, c, h, w = x.size()

        # [Batch_size, channels, Height, Width] -> [Batch_size, channels,       Height * Width]
        x = x.view(n, c, h * w) # each item represents a pixel

        # [Batch_size, channels, Height * Width] -> [Batch_size, Height * Width, channels]
        x = x.transpose(-1, -2)

        # [Batch_size, Height * Width, channels]
        x = self.attention(x)

        # [Batch_size, Height * Width, channels] -> [Batch_size, channels, Height * Width]
        x = x.transpose(-1, -2)

        # [Batch_size, channels, Height * Width] -> [Batch_size, channels, Height, Width]
        x = x.view(n, c, h, w)

        return x + residual




# # 使用 nn.Identity 作为占位符
# 在某些情况下，你可能需要一个“空层”作为占位符，比如调试或动态调整网络架构时，需要保留某些层但暂时不做操作。
# class MyModel(nn.Module):
#     def __init__(self, use_extra_layer=True):
#         super().__init__()
#         self.layer1 = nn.Linear(10, 10)
#         # 如果不需要额外的操作，用 Identity 占位
#         self.extra_layer = nn.Linear(10, 10) if use_extra_layer else nn.Identity()
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.extra_layer(x)
#         return x
#
# model = MyModel(use_extra_layer=False)
# x = torch.randn(1, 10)
# output = model(x)
# print(output)

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAE_ResidualBlock, self).__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [Batch_size, in_channels, Height, Width]

        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super(VAE_Decoder, self).__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512), # [Batch_size, 512, Height / 8, Width / 8]

            nn.Upsample(scale_factor=2), # [Batch_size, 512, Height / 4, Width / 4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2), # [Batch_size, 512, Height / 2, Width / 2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2), # [Batch_size, 256, Height, Width]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1) # [Batch_size, 3, Height, Width]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch_size, 4, Height/8, Width/8]

        # First, we should reverse the scaling (last line in encoder)
        x /= 0.18215

        for module in self:
            x = module(x)

        # [Batch_size, 3, Height, Width]
        return x
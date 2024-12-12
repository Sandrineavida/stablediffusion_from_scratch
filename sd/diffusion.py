import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super(TimeEmbedding, self).__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed) # 320 -> 1280
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed) # 1280 -> 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 320]
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # x: [1, 1280]
        return x

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [Batch_size, channels, Height, Width] -> [Batch_size, channels, Height * 2, Width * 2]
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x



class UNET_ResidualBlock(nn.Module):
    def  __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super(UNET_ResidualBlock, self).__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: [Batch_size, in_channels, Height, Width]
        # time: [1, 1280]

        residual = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residual)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super(UNET_AttentionBlock, self).__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        # GEGLU(x)= x * GELU(x)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x; [Batch_size, Features, Height, Width]
        # context: [Batch_size, seq_len, 768]
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2) # [Batch_size, Height * Width, Features]

        # Normalisation + Self Attention with skip connection
        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        residue_short = x

        # Normalisation + Cross Attention with skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residue_short

        residue_short = x

        # Normalisation + Feed Forward with GEGLU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x = x + residue_short

        # [Batch_size, Height * Width, Features] -> [Batch_size, Features, Height * Width]
        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long



class SwitchSequential(nn.Sequential):
    def __init__(self):
        super(SwitchSequential, self).__init__()
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.encoder = nn.ModuleList([
            # [Batch_size, 4, Height/8, Width/8] -> [Batch_size, 320, Height/8, Width/8]
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            #                                                             n_heads, n_embed

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # [Batch_size, 320, Height/8, Width/8] -> [Batch_size, 320, Height/16, Width/16]
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # [Batch_size, 640, Height/16, Width/16] -> [Batch_size, 640, Height/32, Width/32]
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # [Batch_size, 1280, Height/32, Width/32] -> [Batch_size, 1280, Height/64, Width/64]
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            # [Batch_size, 2 * 1280, Height/64, Width/64] -> [Batch_size, 1280, Height/32, Width/32]
            SwitchSequential(UNET_ResidualBlock(2 * 1280, 1280)), # 2 * 1280 because of the skip connection of unet

            SwitchSequential(UNET_ResidualBlock(2 * 1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(2 * 1280, 1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2 * 1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2 * 1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280),  UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNET_OutputLayer, self).__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch_size, 320, Height/8, Width/8] -> [Batch_size, 4, Height/8, Width/8]
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()

        self.time_embedding = TimeEmbedding(320) # 我们需要把时间戳t也转换为一个embedding，来方便模型理解
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent: [Batch_size, 4, Height/8, Width/8]   output of the VAE encoder
        # context: [Batch_size, seq_len, 768]  output of clip encoder, which is our prompt
        # time: [1, 320] it is like the positional encoding in the transformer model (a combination of sin and cos functions),
        # and it has therefore a similar function as the positional encoding in the transformer model, that is to say,
        # it tells the model, "at which step we arrived in the process of denoisification"

        # [1, 320] -> [1, 1280]
        time = self.time_embedding(time)

        # [Batch_size, 4, Height/8, Width/8] -> [Batch_size, 320, Height/8, Width/8]
        output = self.unet(latent, context, time)

        # [Batch_size, 320, Height/8, Width/8] -> [Batch_size, 4, Height/8, Width/8]
        output = self.final(output)

        # [Batch_size, 4, Height/8, Width/8]
        return output

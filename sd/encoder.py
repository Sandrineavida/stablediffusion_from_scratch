# Encoder part of the Variational Autoencoder
import torch
from torch import nn
from torch.nn import functional as F


from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super(VAE_Encoder, self).__init__(
            # [Batch_size, Channel, Height, Width] -> [Batch_size, 128, Height, Width]
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # [Batch_size, 128, Height, Width] -> [Batch_size, 128, Height, Width]
            VAE_ResidualBlock(128, 128),

            # [Batch_size, 128, Height, Width] -> [Batch_size, 128, Height, Width]
            VAE_ResidualBlock(128, 128),

            # [Batch_size, 128, Height, Width] -> [Batch_size, 128, Height / 2, Width / 2]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # we want to do an asymmetric padding later, so here we set padding=0

            # [Batch_size, 128, Height / 2, Width / 2] -> [Batch_size, 256, Height / 2, Width / 2]
            VAE_ResidualBlock(128, 256),

            # [Batch_size, 256, Height / 2, Width / 2] -> [Batch_size, 256, Height / 2, Width / 2]
            VAE_ResidualBlock(256, 256),

            # [Batch_size, 256, Height / 2, Width / 2] -> [Batch_size, 256, Height / 4, Width / 4]
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), # we want to do an asymmetric padding later, so here we set padding=0

            # [Batch_size, 256, Height / 4, Width / 4] -> [Batch_size, 512, Height / 4, Width / 4]
            VAE_ResidualBlock(256, 512),

            # [Batch_size, 512, Height / 4, Width / 4] -> [Batch_size, 512, Height / 4, Width / 4]
            VAE_ResidualBlock(512, 512),

            # [Batch_size, 512, Height / 4, Width / 4] -> [Batch_size, 512, Height / 8, Width / 8]
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # we want to do an asymmetric padding later, so here we set padding=0

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            VAE_ResidualBlock(512, 512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            VAE_ResidualBlock(512, 512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            VAE_ResidualBlock(512, 512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            VAE_AttentionBlock(512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            VAE_ResidualBlock(512, 512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            nn.GroupNorm(32, 512),

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 512, Height / 8, Width / 8]
            nn.SiLU(), # x * sigmoid(x) = x / (1 + exp(-x))

            # [Batch_size, 512, Height / 8, Width / 8] -> [Batch_size, 8, Height / 8, Width / 8]
            nn.Con2v(512, 8, kernel_size=3, padding=1), # bottleneck

            # [Batch_size, 8, Height / 8, Width / 8] -> [Batch_size, 8, Height / 8, Width / 8]
            nn.Conv2d(8, 8, kernel_size=1, padding=0),

        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x     : [Batch_size, Channel=3,     Height=512, Width=512]
        # noise : [Batch_size, Output_Channels, Height/8, Width/8]
        # Encoder学到的是p(z|x)的分布，即给定输入x，学到的是z的分布,其中，z是一个latent var，p(z|x)是一个高斯分布，由mean和variance表示
        # 得到这个分布之后，我们需要一个确定的数值作为输出，这就意味着我们需要对这个分布进行采样

        for module in self:
            # asymmetric padding
            if getattr(module, 'stride', None) == (2, 2): # if module 具有属性 'stride' 且 module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1)) # # 填充 (左侧0列, 右侧1列, 上方0行, 下方1行)  (padding_left, padding_right, padding_top, padding_bottom)
                # tensor([[1, 2, 3],       -> tensor([[1, 2, 3, 0],
                #         [4, 5, 6],                  [4, 5, 6, 0],
                #         [7, 8, 9]])                 [7, 8, 9, 0],
                #                                     [0, 0, 0, 0]])
            # run sequantielly through all layers/modules
            x = module(x) # we build the latent space X = N(mean, variance)

        # x :       [Batch_size, 8, Height / 8, Width / 8]
        mean, log_variance = torch.chunk(x, 2, dim=1) # devided into 2 tensors along the dimension 1
        # mean    : [Batch_size, 4, Height / 8, Width / 8]
        # log_var : [Batch_size, 4, Height / 8, Width / 8]

        # use clamping (钳制) function to make the log_variance within a certain range
        log_variance =torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp() # make the log_variance to variance (e^ln(x) = x)
        std = variance.sqrt()

        # Now, we're going to sample from the latent space
        # How to sample from the multivariate normal distribution X = N(mean, variance) ? : start with Z = N(0, 1)
        # Z = N(0, 1) -> X = N(mean, variance)
        # => X = mean + std * Z
        x = mean + std * noise

        # Scale the output by a constant from the origianl repo
        x *= 0.18215

        return x



if __name__ == '__main__':
    print("0")

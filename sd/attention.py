import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # for Q, K, V at the same time so "3 * d_model"
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x : [Batch_size, seq_len, d_embed]

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        intermimediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        #                          [Batch_size, seq_len, d_embed]
        # --in_proj-->            [Batch_size, seq_len, 3 * d_embed]
        # --chunk(3, dim = -1)--> 3 * [Batch_size, seq_len, d_embed]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(intermimediate_shape).transpose(1, 2) # [Batch_size, n_heads, seq_len, d_head]
        k = k.view(intermimediate_shape).transpose(1, 2)
        v = v.view(intermimediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        if causal_mask:
            # Mask where the upper triangle (above the pricipal diagonal) is made of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, float('-inf'))

# Attention matrix: [Batch_size, n_heads, seq_len, seq_len]
        A = F.softmax(weight, dim=-1)

        # [Batch_size, n_heads, seq_len, seq_len] @ [Batch_size, n_heads, seq_len, d_head] -> [Batch_size, n_heads, seq_len, d_head]
        # T
        output = A @ v

        # [Batch_size, n_heads, seq_len, d_head] -> [Batch_size, seq_len, n_heads, d_head]
        output = output.transpose(1, 2)

        # [Batch_size, seq_len, n_heads, d_head] -> [Batch_size, seq_len, d_embed]
        output = output.reshape(batch_size, seq_len, d_embed)

        # T x Wo
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super(CrossAttention, self).__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)

        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (latent) : [Batch_size, seq_len_Q, d_embed_Q]
        # y: (context): [Batch_size, seq_len_KV, d_cross_KV] = [Batch_size, 77, 768]
        input_shape = x.shape
        batch_size, seq_len_Q, d_embed_Q = input_shape

        intermimediate_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(intermimediate_shape).transpose(1, 2)
        k = k.view(intermimediate_shape).transpose(1, 2)
        v = v.view(intermimediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        A = F.softmax(weight, dim=-1)

        output = A @ v

        output = output.transpose(1, 2).contiguous()

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

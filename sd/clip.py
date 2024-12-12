import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super(CLIPEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # [Batch_size, seq_len] -> [Batch_size, seq_len, n_embed]
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super(CLIPLayer, self).__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch_size, seq_len, n_embed]
        residual = x

        ## SELF-ATTENTION
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residual

        ## FEED-FORWARD
        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # quick gelu activation fct
        x = self.linear_2(x)
        x += residual

        return x

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77) # vocab size: 49408; embedding size: 768; max seq len: 77

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12) # n_heads: 12; and we have 12 of this layer
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:
        tokens = tokens.type(torch.long)

        # [Batch_size, seq_len] -> [Batch_size, seq_len, 768]
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # [Batch_size, seq_len, 768] -> [Batch_size, seq_len, 768]
        output = self.layernorm(state)

        return output
import torch
from torch import nn
import torch.nn.functional as F

from .config import *


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embedding % n_head == 0, 'reminder should be 0'
        self.c_attn = nn.Linear(n_embedding, 3*n_embedding)
        self.c_proj = nn.Linear(n_embedding, n_embedding)

    def forward(self, x):
        B, T, D = x.shape

        x = self.c_attn(x)
        q, k, v = torch.split(x, n_embedding, dim=-1)

        q = q.view(B, T, n_head, D//n_head).transpose(1, 2)
        k = k.view(B, T, n_head, D//n_head).transpose(1, 2)
        v = v.view(B, T, n_head, D//n_head).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).view(B, T, D)

        return self.c_proj(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embedding, 4*n_embedding)
        self.c_proj = nn.Linear(4*n_embedding, n_embedding)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embedding)
        self.ln_2 = nn.LayerNorm(n_embedding)
        self.attn = MultiheadAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(n_vocab, n_embedding),
            wpe=nn.Embedding(max_sequence_len, n_embedding),
            h=nn.ModuleList([Block() for _ in range(n_block)]),
            ln_f=nn.LayerNorm(n_embedding),
        ))
        self.lm_head = nn.Linear(n_embedding, n_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.smart_init()

    def smart_init(self):
        def initer(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
        self.apply(initer)
        # residual compensation
        std = 0.02 * (2 * n_block) ** -0.5
        for block in self.transformer.h:
            nn.init.normal_(block.attn.c_proj.weight, std=std)
            nn.init.normal_(block.mlp.c_proj.weight, std=std)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        assert T <= max_sequence_len, 'sequence too long'

        index = torch.arange(T).to(device)
        x = self.transformer.wte(x)  # => B, T, D
        pe = self.transformer.wpe(index)  # => T, D
        x = x + pe  # => B, T, D

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return self.lm_head(x)


if __name__ == '__main__':
    x = torch.zeros(3, 5).long().to(device)
    print(GPT().to(device)(x).shape)

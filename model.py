from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    width: int = 128
    heads: int = 4
    layers: int = 4
    context: int = 128

    def __post_init__(self):
        if min(self.vocab_size, self.width, self.heads, self.layers, self.context) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.ff(self.ln2(x))


class TinyLLM(nn.Module):
    def __init__(self, vocab_size=256, width=128, heads=4, layers=4, context=128):
        super().__init__()
        self.config = ModelConfig(vocab_size, width, heads, layers, context)
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(self.config) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        length = ids.size(1)
        if not 0 < length <= self.context:
            raise ValueError("Input must contain between one token and the model context length")
        positions = torch.arange(length, device=ids.device)
        hidden = self.token(ids) + self.position(positions)[None, :, :]
        logits = self.output(self.norm(self.blocks(hidden)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

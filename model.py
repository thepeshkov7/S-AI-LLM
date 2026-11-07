import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        z = self.ln1(x)
        n = z.size(1)
        mask = torch.ones(n, n, device=z.device, dtype=torch.bool).triu(1)
        z, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        x = x + z
        return x + self.ff(self.ln2(x))


class TinyLLM(nn.Module):
    def __init__(self, vocab_size=256, width=128, heads=4, layers=4, context=128):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, ids, targets=None):
        b, t = ids.shape
        if t > self.context:
            raise ValueError("Input exceeds context length")
        pos = torch.arange(t, device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        logits = self.output(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

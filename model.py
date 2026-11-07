from dataclasses import dataclass
import math

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
    architecture = None

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

    def _encode(self, ids):
        length = ids.size(1)
        if not 0 < length <= self.context:
            raise ValueError("Input must contain between one token and the model context length")
        positions = torch.arange(length, device=ids.device)
        hidden = self.token(ids) + self.position(positions)[None, :, :]
        return self.norm(self.blocks(hidden))

    def forward(self, ids, targets=None):
        logits = self.output(self._encode(ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

    def answer_loss(self, ids, targets):
        """Compute the language-model loss only at supervised target positions."""
        active = targets != -100
        return F.cross_entropy(self.output(self._encode(ids)[active]), targets[active])

    def next_scores(self, ids, positions=None):
        scores, _ = self(ids)
        if positions is None:
            return scores[:, -1]
        return scores[torch.arange(len(ids), device=ids.device), positions]


class CopyLLM(TinyLLM):
    """Learn when to generate a token or copy it from earlier positions."""

    architecture = "copy_v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.config.width
        self.copy_width = min(32, width)
        self.copy_query = nn.Linear(width, self.copy_width, bias=False)
        self.copy_key = nn.Linear(width, self.copy_width, bias=False)
        self.copy_mode = nn.Linear(width, 3)
        self.register_buffer("copy_aliases", torch.arange(self.config.vocab_size))
        for projection in (self.copy_query, self.copy_key, self.copy_mode):
            self._initialize(projection)
        with torch.no_grad():
            # Start close to the loaded vocabulary head.
            self.copy_mode.bias.copy_(torch.tensor([3.0, -1.0, -1.0]))

    def set_copy_aliases(self, tokenizer):
        """Allow a copied byte-level BPE token with or without its leading space."""
        vocab = tokenizer.get_vocab()
        pairs = {vocab[token[1:]]: index for token, index in vocab.items()
                 if token.startswith("Ġ") and not token.startswith("ĠĠ") and token[1:] in vocab}
        aliases = torch.arange(len(vocab))
        for plain, spaced in pairs.items():
            aliases[plain], aliases[spaced] = spaced, plain
        self.copy_aliases.copy_(aliases)

    def _copy_attention(self, query_states, hidden, positions):
        keys = self.copy_key(hidden)
        queries = self.copy_query(query_states)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.copy_width)
        visible = torch.arange(hidden.size(1), device=hidden.device) <= positions.unsqueeze(-1)
        return scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)

    def _mixture(self, states, source_ids, attention):
        mode = self.copy_mode(states).softmax(dim=-1)
        probabilities = self.output(states).softmax(dim=-1) * mode[..., :1]
        probabilities = probabilities.scatter_add(-1, source_ids, attention * mode[..., 1:2])
        probabilities = probabilities.scatter_add(-1, self.copy_aliases[source_ids], attention * mode[..., 2:3])
        return probabilities.clamp_min(1e-30).log()

    def forward(self, ids, targets=None):
        hidden = self._encode(ids)
        positions = torch.arange(ids.size(1), device=ids.device)
        attention = self._copy_attention(hidden, hidden, positions)
        source = ids[:, None, :].expand_as(attention)
        scores = self._mixture(hidden, source, attention)
        loss = None if targets is None else F.nll_loss(scores.flatten(0, 1), targets.flatten())
        return scores, loss

    def next_scores(self, ids, positions=None):
        if positions is None:
            positions = ids.new_full((len(ids),), ids.size(1) - 1)
        hidden = self._encode(ids)
        states = hidden[torch.arange(len(ids), device=ids.device), positions]
        attention = self._copy_attention(states[:, None], hidden, positions[:, None])[:, 0]
        return self._mixture(states, ids, attention)

    def answer_loss(self, ids, targets):
        hidden = self._encode(ids)
        positions = torch.arange(ids.size(1), device=ids.device)
        rows, ends = (targets != -100).nonzero(as_tuple=True)
        states = hidden[rows, ends]
        attention = self._copy_attention(hidden, hidden, positions)[rows, ends]
        source = ids[rows]
        target = targets[rows, ends, None]
        vocab = self.output(states).log_softmax(dim=-1).gather(-1, target)
        literal = (attention * (source == target)).sum(dim=-1, keepdim=True)
        alias = (attention * (self.copy_aliases[source] == target)).sum(dim=-1, keepdim=True)
        terms = torch.cat((vocab, literal.clamp_min(1e-30).log(), alias.clamp_min(1e-30).log()), dim=-1)
        return -(terms + self.copy_mode(states).log_softmax(dim=-1)).logsumexp(dim=-1).mean()

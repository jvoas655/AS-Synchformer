"""
Mostly taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier

Mods:
- we use it as the encoder so it was slightly modified to avoid confusion with language modelling (GPT)
- we “unify” the different attention implementations into a single MultiHeadAttention (and a trivial CrossAttention subclass)
  and update the Blocks accordingly. (The ones used by CrossBlock and OldBlock are preserved.)
- the new Block still supports soft masking.
"""

import logging
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return False


def apply_rope(q, k, seq_len, dim_head, device):
    # Generate RoPE sinusoidal positional embeddings
    position = torch.arange(seq_len, dtype=torch.float32).to(device)  # 1D tensor of positions
    div_term = torch.exp(torch.arange(0, dim_head, 2, dtype=torch.float32).to(device) *
                         -(math.log(10000.0) / dim_head))
    sinusoid = torch.einsum('i,j->ij', position, div_term)  # Shape (seq_len, dim_head//2)
    sin, cos = sinusoid.sin(), sinusoid.cos()

    # Split q and k into even and odd parts
    q_odd, q_even = q[..., ::2], q[..., 1::2]
    k_odd, k_even = k[..., ::2], k[..., 1::2]

    sin = sin.unsqueeze(0).unsqueeze(1)  # Shape (1, 1, seq_len, dim_head//2)
    cos = cos.unsqueeze(0).unsqueeze(1)

    q_rotated = torch.cat([
        q_even * cos - q_odd * sin,
        q_even * sin + q_odd * cos
    ], dim=-1)

    k_rotated = torch.cat([
        k_even * cos - k_odd * sin,
        k_even * sin + k_odd * cos
    ], dim=-1)

    return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    """
    Unified multi-head attention module.
    
    If a 'context' tensor is provided in forward(), the module performs cross-attention.
    Otherwise it defaults to self-attention.
    
    Optional soft masking is applied if soft_mask is not None.
    An optional RoPE rotation can be applied (e.g. for language modeling).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        bias = getattr(config, "bias", True)

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=bias)
        self.key   = nn.Linear(config.n_embd, config.n_embd, bias=bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=bias)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=bias)

    def forward(self, x, soft_mask=None, context=None, use_rope=False, return_attn_weights=False):
        """
        Args:
            x: input tensor of shape (B, T, C)
            soft_mask: optional mask of shape (B, T, T) [or (B, T, T_context) for cross-attention]
            context: if provided, must be of shape (B, T_context, C) for cross-attention.
            use_rope: if True, apply RoPE to q and k.
            return_attn_weights: if True, also return attention weights.
        """
        B, T, C = x.size()
        # Compute query.
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        
        if context is None:
            # Self-attention: key and value from x.
            k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        else:
            # Cross-attention: key and value from context.
            B_ctx, T_ctx, C_ctx = context.size()
            assert B_ctx == B and C_ctx == C, "Context must have same batch size and embedding dim as x."
            k = self.key(context).view(B, T_ctx, self.n_head, self.head_dim).transpose(1, 2)
            v = self.value(context).view(B, T_ctx, self.n_head, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = apply_rope(q, k, seq_len=T, dim_head=self.head_dim, device=x.device)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T) or (B, n_head, T, T_ctx)

        if soft_mask is not None:
            # Ensure soft_mask has shape (B, 1, T, T) or (B, 1, T, T_ctx)
            soft_mask = soft_mask.view(B, 1, att.size(-2), att.size(-1))
            max_attn = att.max(dim=-1, keepdim=True)[0].detach()
            min_attn = att.min(dim=-1, keepdim=True)[0].detach()
            # In training, scale by half the range; else use a large constant.
            range_attn = (max_attn - min_attn) if self.training else 1e9
            scaling_factor = 1 - soft_mask
            att = att - scaling_factor * range_attn

        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_drop(att_weights)
        y = att_weights @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        
        if return_attn_weights:
            return y, att_weights
        return y


class CrossAttention(MultiHeadAttention):
    """
    A trivial subclass of MultiHeadAttention for cross-attention.
    Its forward() simply requires a context tensor.
    """
    def forward(self, x, context, soft_mask=None, use_rope=False, return_attn_weights=False):
        return super().forward(x, soft_mask=soft_mask, context=context,
                               use_rope=use_rope, return_attn_weights=return_attn_weights)


class Block(nn.Module):
    """
    A transformer block that uses the unified MultiHeadAttention.
    Supports an optional soft mask and an optional no-residual mode.
    """
    def __init__(self, config, skip_mlp=False):
        super().__init__()
        bias = getattr(config, "bias", True)
        self.force_rope = False
        self.skip_mlp = skip_mlp
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        if not self.skip_mlp:
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=bias),
                nn.Dropout(config.resid_pdrop),
            )

    def forward(self, x, soft_mask=None, use_rope=False, no_resid=False, return_attn_weights=False):
        # Force RoPE if desired.
        actual_use_rope = use_rope or self.force_rope
        if return_attn_weights:
            attn_out, attn_weights = self.attn(self.ln1(x), soft_mask=soft_mask,
                                               use_rope=actual_use_rope, return_attn_weights=True)
            if no_resid:
                x = self.mlp(self.ln2(attn_out)) if not self.skip_mlp else attn_out
                return x, attn_weights
            else:
                x = x + attn_out
                if not self.skip_mlp:
                    x = x + self.mlp(self.ln2(x))
                return x, attn_weights
        else:
            if no_resid:
                x = self.attn(self.ln1(x), soft_mask=soft_mask, use_rope=actual_use_rope)
                if not self.skip_mlp:
                    x = self.mlp(self.ln2(x))
                return x
            else:
                x = x + self.attn(self.ln1(x), soft_mask=soft_mask, use_rope=actual_use_rope)
                if not self.skip_mlp:
                    x = x + self.mlp(self.ln2(x))
                return x



class CrossBlock(nn.Module):
    """
    Similar to transformer.Block (an encoder layer), but with cross-attention (as in decoder layers).
    Uses self-attention (via MultiHeadAttention) and cross-attention (via CrossAttention).
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)  # self-attention
        self.xattn = CrossAttention(config)       # cross-attention
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x_tuple, soft_mask=None, return_attn_weights=False):
        """
        Args:
            x_tuple: a tuple (x, context)
            soft_mask: optional soft mask (if provided, for both self- and cross-attention)
            return_attn_weights: if True, return attention weights.
        """
        x, context = x_tuple
        # For self-attention, if a soft mask is provided, use only the square part.
        if soft_mask is not None:
            T = min(soft_mask.shape[-1], soft_mask.shape[-2])
            self_attn_mask = soft_mask[..., :T, :T]
        else:
            self_attn_mask = None

        if return_attn_weights:
            cross_out, cross_attn = self.xattn(self.ln2(x), context, soft_mask=soft_mask,
                                                use_rope=False, return_attn_weights=True)
            x = x + cross_out
            self_out, self_attn = self.attn(self.ln1(x), soft_mask=self_attn_mask,
                                            use_rope=False, return_attn_weights=True)
            x = x + self_out
        else:
            x = x + self.xattn(self.ln2(x), context, soft_mask=soft_mask)
            x = x + self.attn(self.ln1(x), soft_mask=self_attn_mask)
        x = x + self.mlp(self.ln3(x))
        if return_attn_weights:
            return x, context, self_attn, cross_attn
        return x, context


# ---------------------------------------------------------------------
# The remaining classes (positional encodings, normalization, and the full Transformer)
# are kept unchanged.
# ---------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, n_embd, pos_dim=None, max_period=200):
        """
        Args:
            n_embd (int): The full model embedding dimension.
            pos_dim (int): The dimension used for the sinusoidal embedding.
                           Defaults to n_embd // 4.
            max_period (float): The maximum period (similar to the 10000 in the original Transformer).
        """
        super().__init__()
        self.n_embd = n_embd
        self.pos_dim = pos_dim if pos_dim is not None else n_embd // 4
        self.max_period = max_period
        self.proj = nn.Linear(self.pos_dim, self.n_embd)

    def forward(self, x, spacing=None):
        """
        Args:
            x (Tensor): A tensor of shape (B, L, D) whose positions we are going to add.
                        (Typically this is the concatenation of the off token and tokens.)
            spacing (Tensor, optional): A tensor of shape (B, L-1) giving the separation between
                        successive token positions (not including the off token, which is always at 0).
                        If None, we assume a constant spacing of 1.
        Returns:
            pos_emb_projected (Tensor): A tensor of shape (B, L, n_embd) to be added to x.
        """
        B, L, _ = x.shape
        device = x.device

        if spacing is None:
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L).float()
        else:
            spacing = spacing.float().to(device)
            pos = torch.cat([torch.zeros(B, 1, device=device), torch.cumsum(spacing, dim=1)], dim=1)
        
        half_dim = self.pos_dim // 2
        exponent = 2 * torch.arange(half_dim, device=device).float() / self.pos_dim
        inv_freq = 1.0 / (self.max_period ** exponent)
        angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        pos_emb = torch.cat([sin_part, cos_part], dim=-1)
        pos_emb_projected = self.proj(pos_emb)
        return pos_emb_projected

class NoPosEncoding(nn.Module):
    """Does not apply any positional encoding"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x


class ZeroInitPositionalEncoding(nn.Module):
    """Zero-initialized trainable positional embedding."""
    def __init__(self, block_shape, n_embd):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.zeros(1, *block_shape, n_embd))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb


class RandInitPositionalEncoding(nn.Module):
    """Random-initialized trainable positional embedding."""
    def __init__(self, block_shape: list, n_embd: int):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.randn(1, *block_shape, n_embd))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb


class PositionEmbeddingLearnedVisual(nn.Module):
    def __init__(self, block_shape, n_embd) -> None:
        super().__init__()
        self.block_shape = block_shape
        self.max_t, self.max_h, self.max_w = block_shape
        self.n_embd = n_embd
        self.n_dims = len(block_shape)
        self.n_embd_t = self.n_embd_h = self.n_embd_w = self.n_embd // self.n_dims
        self.n_embd_t += self.n_embd % self.n_dims
        self.time_embed = nn.Embedding(self.max_t, self.n_embd_t)
        self.height_embed = nn.Embedding(self.max_h, self.n_embd_h)
        self.width_embed = nn.Embedding(self.max_w, self.n_embd_w)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)
        nn.init.uniform_(self.height_embed.weight)
        nn.init.uniform_(self.width_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a batch of visual feature maps (B, t, h, w, d)
        Returns:
            torch.Tensor: x + positional embeddings
        """
        return x + self.make_pos_emb(x)

    def make_pos_emb(self, x):
        B, t, h, w, d = x.shape
        t_i = torch.arange(t, device=x.device)
        h_i = torch.arange(h, device=x.device)
        w_i = torch.arange(w, device=x.device)
        t_emb = self.time_embed(t_i).view(t, 1, 1, self.n_embd_t).repeat(1, h, w, 1)
        h_emb = self.height_embed(h_i).view(1, h, 1, self.n_embd_h).repeat(t, 1, w, 1)
        w_emb = self.width_embed(w_i).view(1, 1, w, self.n_embd_w).repeat(t, h, 1, 1)
        pos = torch.cat([t_emb, h_emb, w_emb], dim=-1)
        pos = pos.view(1, t, h, w, d).repeat(B, 1, 1, 1, 1)
        return pos


class PositionEmbeddingLearnedAudio(nn.Module):
    def __init__(self, block_shape, n_embd) -> None:
        super().__init__()
        self.block_shape = block_shape
        self.max_f, self.max_t = block_shape
        self.n_dims = len(block_shape)
        self.n_embd_t = self.n_embd_f = n_embd // self.n_dims
        self.n_embd_t += n_embd % self.n_dims
        self.freq_embed = nn.Embedding(self.max_f, self.n_embd_f)
        self.time_embed = nn.Embedding(self.max_t, self.n_embd_t)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.freq_embed.weight)
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a batch of spectrogram feature maps (B, f, t, d)
        Returns:
            torch.Tensor: x + positional embeddings
        """
        return x + self.make_pos_emb(x)

    def make_pos_emb(self, x):
        B, f, t, d = x.shape
        f_i = torch.arange(f, device=x.device)
        t_i = torch.arange(t, device=x.device)
        f_emb = self.freq_embed(f_i).view(f, 1, self.n_embd_f).repeat(1, t, 1)
        t_emb = self.time_embed(t_i).view(1, t, self.n_embd_t).repeat(f, 1, 1)
        pos = torch.cat([f_emb, t_emb], dim=-1)
        pos = pos.view(1, f, t, d).repeat(B, 1, 1, 1)
        return pos


class L2Normalize(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        eps = 1e-6 if x.dtype in [torch.half, torch.float16] else 1e-12
        return F.normalize(x, p=2.0, dim=-1, eps=eps)

"""
MoReFun architecture.

Reconstructs the model from the paper's equations (Sec. 3.2 main + Sec. 7
supplementary). The two main modules are:

    PastMotionEncoder (PME)
        - 3 stacked blocks
        - each block: temporal self-attn -> FFN -> spatial self-attn -> FFN

    FutureMotionDecoder (FMD)
        - 3 stacked blocks
        - each block: temporal cross-attn -> FFN -> spatial cross-attn -> FFN
                     -> temporal self-attn -> FFN -> spatial self-attn -> FFN

All attention is multi-head scaled dot-product. Time and joint axes are
handled by reshaping: temporal attention puts joints in the batch axis and
attends over time; spatial attention does the reverse.

The spatial cross-attention has an ambiguity in the paper (Q has L frames,
K/V have T frames, but spatial attention is along the J axis). We resolve it
by mean-pooling the past along its time axis to get a (B, J, C) summary,
which is then used as K/V for every future frame's spatial cross-attention.
This matches the joint-by-joint attention maps shown in the paper's Fig. 12(d).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Position-wise FFN ("bottle neck" in the paper).

    Two linears with GELU between them, plus residual and LayerNorm.
    """

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with separate Q and K/V inputs.

    Used as both self-attention (q == kv) and cross-attention (q != kv).
    Includes residual on the query side and a final LayerNorm.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_in: (B, Nq, C)
            kv_in: (B, Nk, C)

        Returns:
            (B, Nq, C)
        """
        B, Nq, _ = q_in.shape
        Nk = kv_in.shape[1]

        q = self.q_proj(q_in).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, Nq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.num_heads * self.head_dim)
        out = self.proj_dropout(self.out_proj(out))

        return self.norm(q_in + out)


# ---------------------------------------------------------------------------
# Reshaping helpers
# ---------------------------------------------------------------------------


def fold_joints_into_batch(x: torch.Tensor) -> torch.Tensor:
    """(B, T, J, C) -> (B*J, T, C). Joints become parallel batch entries."""
    B, T, J, C = x.shape
    return x.permute(0, 2, 1, 3).reshape(B * J, T, C)


def unfold_joints_from_batch(x: torch.Tensor, B: int, J: int) -> torch.Tensor:
    """(B*J, T, C) -> (B, T, J, C)."""
    BJ, T, C = x.shape
    assert BJ == B * J
    return x.view(B, J, T, C).permute(0, 2, 1, 3).contiguous()


def fold_time_into_batch(x: torch.Tensor) -> torch.Tensor:
    """(B, T, J, C) -> (B*T, J, C). Time becomes parallel batch entries."""
    B, T, J, C = x.shape
    return x.reshape(B * T, J, C)


def unfold_time_from_batch(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """(B*T, J, C) -> (B, T, J, C)."""
    BT, J, C = x.shape
    assert BT == B * T
    return x.view(B, T, J, C)


# ---------------------------------------------------------------------------
# PME and FMD blocks
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    """
    One PME block: temporal self-attn -> FFN -> spatial self-attn -> FFN.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_attn = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.t_ffn = FeedForward(dim, ffn_mult, dropout)
        self.s_attn = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.s_ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, C)
        B, T, J, C = x.shape

        # Temporal self-attention: joints batched, time attended.
        h = fold_joints_into_batch(x)  # (B*J, T, C)
        h = self.t_attn(h, h)
        h = self.t_ffn(h)
        x = unfold_joints_from_batch(h, B, J)  # (B, T, J, C)

        # Spatial self-attention: time batched, joints attended.
        h = fold_time_into_batch(x)  # (B*T, J, C)
        h = self.s_attn(h, h)
        h = self.s_ffn(h)
        x = unfold_time_from_batch(h, B, T)  # (B, T, J, C)

        return x


class DecoderBlock(nn.Module):
    """
    One FMD block: 4 attention sublayers in sequence.

      temporal cross-attn -> FFN
      spatial cross-attn  -> FFN
      temporal self-attn  -> FFN
      spatial self-attn   -> FFN

    The cross-attentions read from the past (H_P); the self-attentions
    let future tokens coordinate with each other.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_cross = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.t_cross_ffn = FeedForward(dim, ffn_mult, dropout)

        self.s_cross = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.s_cross_ffn = FeedForward(dim, ffn_mult, dropout)

        self.t_self = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.t_self_ffn = FeedForward(dim, ffn_mult, dropout)

        self.s_self = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.s_self_ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(self, h_f: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_f: future state, (B, L, J, C)
            h_p: past features, (B, T, J, C)
        """
        B, L, J, C = h_f.shape
        T = h_p.shape[1]

        # --- Temporal cross-attention ---
        # Queries: future folded to (B*J, L, C). Keys/values: past folded to (B*J, T, C).
        q = fold_joints_into_batch(h_f)
        kv = fold_joints_into_batch(h_p)
        h = self.t_cross(q, kv)
        h = self.t_cross_ffn(h)
        h_f = unfold_joints_from_batch(h, B, J)  # (B, L, J, C)

        # --- Spatial cross-attention ---
        # The past has T frames, the future query is per-frame. We resolve the
        # mismatch by mean-pooling the past along its time axis to a per-joint
        # summary, then doing per-future-frame spatial attention against it.
        h_p_pooled = h_p.mean(dim=1, keepdim=True)  # (B, 1, J, C)
        # Broadcast pooled past across L future frames for batching.
        h_p_pooled_b = h_p_pooled.expand(-1, L, -1, -1)  # (B, L, J, C)

        q = fold_time_into_batch(h_f)              # (B*L, J, C)
        kv = fold_time_into_batch(h_p_pooled_b)    # (B*L, J, C)
        h = self.s_cross(q, kv)
        h = self.s_cross_ffn(h)
        h_f = unfold_time_from_batch(h, B, L)  # (B, L, J, C)

        # --- Temporal self-attention ---
        h = fold_joints_into_batch(h_f)  # (B*J, L, C)
        h = self.t_self(h, h)
        h = self.t_self_ffn(h)
        h_f = unfold_joints_from_batch(h, B, J)  # (B, L, J, C)

        # --- Spatial self-attention ---
        h = fold_time_into_batch(h_f)  # (B*L, J, C)
        h = self.s_self(h, h)
        h = self.s_self_ffn(h)
        h_f = unfold_time_from_batch(h, B, L)  # (B, L, J, C)

        return h_f


# ---------------------------------------------------------------------------
# Joint embedder, encoder, decoder, full model
# ---------------------------------------------------------------------------


class JointEmbedder(nn.Module):
    """
    Linear lift from coordinates to features, plus trainable temporal and
    spatial position encodings (Eq. 1 of the paper).

    Position encodings are stored at the maximum sequence length the model
    ever sees. The forward pass crops them to the actual T or L of the input.
    """

    def __init__(self, coord_dim: int, num_joints: int, channels: int,
                 max_len: int = 64):
        super().__init__()
        self.linear = nn.Linear(coord_dim, channels)
        self.pos_t = nn.Parameter(torch.zeros(max_len, channels))
        self.pos_s = nn.Parameter(torch.zeros(num_joints, channels))
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_s, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, K)
        Returns:
            (B, T, J, C)
        """
        B, T, J, _ = x.shape
        h = self.linear(x)  # (B, T, J, C)
        h = h + self.pos_t[:T].view(1, T, 1, -1)
        h = h + self.pos_s[:J].view(1, 1, J, -1)
        return h


class PastMotionEncoder(nn.Module):
    """
    PME: stack of EncoderBlocks.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 num_blocks: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, num_heads, head_dim, ffn_mult, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class FutureMotionDecoder(nn.Module):
    """
    FMD: stack of DecoderBlocks.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 num_blocks: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, num_heads, head_dim, ffn_mult, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, h_f: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h_f = blk(h_f, h_p)
        return h_f


class MoReFun(nn.Module):
    """
    Full model: shared joint embedder, PME, FMD, and a linear prediction head.

    Used in three modes:
      - pretrain past:   encode masked past, decode back to coordinates
      - pretrain future: encode full past, decode masked future
      - finetune/predict: encode full past, decode zero-future to predict full future

    The text branch of the paper is omitted here since we are training on
    Human3.6M, which has no text annotations.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.model

        max_len = max(cfg.data.past_len, cfg.data.future_len) + 4

        self.embed = JointEmbedder(
            coord_dim=cfg.data.coord_dim,
            num_joints=cfg.data.num_joints,
            channels=d.channels,
            max_len=max_len,
        )
        self.encoder = PastMotionEncoder(
            dim=d.channels,
            num_heads=d.num_heads,
            head_dim=d.head_dim,
            num_blocks=d.num_blocks,
            ffn_mult=d.ffn_mult,
            dropout=d.dropout,
        )
        self.decoder = FutureMotionDecoder(
            dim=d.channels,
            num_heads=d.num_heads,
            head_dim=d.head_dim,
            num_blocks=d.num_blocks,
            ffn_mult=d.ffn_mult,
            dropout=d.dropout,
        )
        self.head = nn.Linear(d.channels, cfg.data.coord_dim)

    # -- Building-block forward passes --

    def encode_past(self, x_past: torch.Tensor) -> torch.Tensor:
        """(B, T, J, K) -> (B, T, J, C). Used for both masked and full past."""
        return self.encoder(self.embed(x_past))

    def decode_future(self, x_future: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        """
        Decode a future input (zeros at fine-tune, masked at pretrain) given past features.

        Args:
            x_future: (B, L, J, K)
            h_p: (B, T, J, C)

        Returns:
            (B, L, J, K) reconstructed/predicted future coordinates.
        """
        h_f = self.embed(x_future)
        h_f = self.decoder(h_f, h_p)
        return self.head(h_f)

    # -- Task-specific forward passes --

    def reconstruct_past(self, x_past_masked: torch.Tensor) -> torch.Tensor:
        """
        Pretraining task 1: reconstruct the past from a masked version of itself.

        Args:
            x_past_masked: (B, T, J, K)
        Returns:
            (B, T, J, K)
        """
        h = self.encode_past(x_past_masked)
        return self.head(h)

    def reconstruct_future(self, x_past_full: torch.Tensor,
                           x_future_masked: torch.Tensor) -> torch.Tensor:
        """
        Pretraining task 2: reconstruct the future given the complete past
        and a partially masked future.

        Args:
            x_past_full: (B, T, J, K) -- not masked
            x_future_masked: (B, L, J, K)
        Returns:
            (B, L, J, K)
        """
        h_p = self.encode_past(x_past_full)
        return self.decode_future(x_future_masked, h_p)

    def predict_future(self, x_past_full: torch.Tensor) -> torch.Tensor:
        """
        Fine-tune / inference: predict the future from the past.

        The future input is a tensor of zeros that picks up only the position
        encodings inside the joint embedder.

        Args:
            x_past_full: (B, T, J, K)
        Returns:
            (B, L, J, K)
        """
        B = x_past_full.shape[0]
        L = self.embed.pos_t.shape[0]  # capacity
        # We need to know the future length. Pass it in via a zero tensor that
        # the caller shapes correctly. Here we expose a wrapper instead:
        raise NotImplementedError("Use forward(...) which accepts future_len.")

    def forward(self, x_past: torch.Tensor, future_len: int) -> torch.Tensor:
        """
        Standard prediction call: past -> future of length future_len.
        """
        B, T, J, K = x_past.shape
        zeros_future = torch.zeros(B, future_len, J, K, device=x_past.device,
                                   dtype=x_past.dtype)
        h_p = self.encode_past(x_past)
        return self.decode_future(zeros_future, h_p)

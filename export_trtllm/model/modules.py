from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
import numpy as np
import tensorrt as trt
from ..._utils import (fp32_array, int32_array, is_same_dtype, set_obj_attrs,
                      trt_dtype_to_np, trt_dtype_to_str,str_dtype_to_trt)
from ...functional import (Tensor, allgather, arange, chunk, concat, constant,
                           cos, exp, expand, shape, silu, sin, slice, split, permute,
                           unsqueeze, matmul, softmax, where, RopeEmbeddingUtils, minimum, repeat_interleave, squeeze, cast, gelu)
from ...functional import expand_dims, view
from ...layers import MLP, BertAttention, Conv2d, LayerNorm, Linear, Conv1d, Mish, embedding, RowLinear, ColumnLinear
from ...module import Module, ModuleList

class GRN(Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class FeedForward(Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.approximate = approximate
        self.project_in = Linear(dim, inner_dim)
        self.ff = Linear(inner_dim, dim_out)

    def forward(self, x):
        return self.ff(gelu(self.project_in(x)))

class ConvNeXtV2Block(Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        unsqueeze(start, 1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)

class AdaLayerNormZero(Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = Linear(dim, dim * 6)
        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(emb, 6, dim=1)
        x = self.norm(x)
        ones = constant(np.ones(1, dtype = np.float32)).cast(x.dtype)
        x = x * (ones + unsqueeze(scale_msa, 1)) + unsqueeze(shift_msa, 1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNormZero_Final(Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = Linear(dim, dim * 2)

        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(silu(emb))
        scale, shift = chunk(emb, 2, dim=1)
        # scale ----> (1, 1024)
        # x     ----> (1, -1, 1024)
        ones = constant(np.ones(1, dtype = np.float32)).cast(x.dtype)
        x = self.norm(x) * unsqueeze((ones + scale), 1)
        x = x + unsqueeze(shift, 1)
        return x

class ConvPositionEmbedding(Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d1 = Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.conv1d2 = Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.mish = Mish()

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = permute(x, [0, 2, 1])
        x = self.mish(self.conv1d2(self.mish(self.conv1d1(x))))
        out = permute(x, [0, 2, 1])
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out

class Attention(Module):
    def __init__(
        self,
        processor: AttnProcessor,
        dim: int,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None, # if not None -> joint attention
        context_pre_only = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim # hidden_size
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.attention_head_size = dim_head
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        self.tp_size = 1
        self.num_attention_heads = heads // self.tp_size
        self.num_attention_kv_heads = heads // self.tp_size # 8
        self.dtype = str_dtype_to_trt('float32')
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        # self.to_q = Linear(dim, self.inner_dim)
        self.to_q = ColumnLinear(dim,
                              self.tp_size * self.num_attention_heads *self.attention_head_size,
                               bias=True,
                               dtype=self.dtype,
                               tp_group=None,
                               tp_size=self.tp_size)
        self.to_k = ColumnLinear(dim,
                              self.tp_size * self.num_attention_heads *self.attention_head_size,
                               bias=True,
                               dtype=self.dtype,
                               tp_group=None,
                               tp_size=self.tp_size)
        self.to_v = ColumnLinear(dim,
                              self.tp_size * self.num_attention_heads *self.attention_head_size,
                               bias=True,
                               dtype=self.dtype,
                               tp_group=None,
                               tp_size=self.tp_size)

        if self.context_dim is not None:
            self.to_k_c = Linear(context_dim, self.inner_dim)
            self.to_v_c = Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = Linear(context_dim, self.inner_dim)

        # self.to_out = Linear(self.inner_dim, dim)
        self.to_out = RowLinear(self.tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               dim,
                               bias=True,
                               dtype=self.dtype,
                               tp_group=None,
                               tp_size=self.tp_size)
        # self.to_out.append(Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = Linear(self.inner_dim, dim)


    def forward(
            self,
            x: float['b n d'],  # noised input x
            rope_cos,
            rope_sin,
            c: float['b n d'] = None,  # context c
            mask: bool['b n'] | None = None,
            scale = 1.0,
            rope=None,
            c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, scale=scale, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, rope_cos=rope_cos, rope_sin=rope_sin, mask=mask, scale=scale)

def rotate_every_two_3dim(tensor: Tensor) -> Tensor:
    assert tensor.ndim() == 3

    shape_tensor = concat([
        shape(tensor, i) / 2 if i == (tensor.ndim() -
                                      1) else shape(tensor, i)
        for i in range(tensor.ndim())
    ])
    x1 = slice(tensor, [0, 0,  0], shape_tensor, [1, 1, 2])
    x2 = slice(tensor, [0, 0,  1], shape_tensor, [1, 1, 2])
    x1 = expand_dims(x1, 3)
    x2 = expand_dims(x2, 3)
    zero = constant(
        np.ascontiguousarray(
            np.zeros([1], dtype=trt_dtype_to_np(tensor.dtype))))
    x2 = zero - x2
    x = concat([x2, x1], 3)
    out =  view(
        x, concat([shape(x, 0),
                   shape(x, 1),
                   shape(x, 2)*2]))

    return out

def apply_rotary_pos_emb_3dim(x, rope_cos, rope_sin):
    rot_dim = shape(rope_cos, 2) #64
    new_t_shape =  concat([shape(x, 0), shape(x, 1), rot_dim]) # (2, -1, 64)
    x_ = slice(x, [0, 0, 0], new_t_shape, [1, 1, 1])
    end_dim = shape(x, 2) - shape(rope_cos, 2)
    new_t_unrotated_shape = concat([shape(x, 0), shape(x, 1), end_dim]) # (2, -1, 960)
    x_unrotated = slice(x, concat([0, 0, rot_dim]), new_t_unrotated_shape, [1, 1, 1])
    out = concat([x_ * rope_cos + rotate_every_two_3dim(x_) * rope_sin, x_unrotated], dim = -1)
    # t -> (2,-1,1024)   freqs -> (-1,64)
    return out

class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float['b n d'],  # noised input x
        rope_cos,
        rope_sin,
        mask: bool['b n'] | None = None,
        scale = 1.0,
        rope=None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        # k,v,q all (2,1226,1024)
        query = apply_rotary_pos_emb_3dim(query, rope_cos, rope_sin)
        key = apply_rotary_pos_emb_3dim(key, rope_cos, rope_sin)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def transpose_for_scores(x):
            new_x_shape = concat([
                batch_size,
                -1, attn.num_attention_heads, attn.attention_head_size
            ])

            y = x.view(new_x_shape)
            y = y.transpose(1, 2)
            return y

        def transpose_for_scores_k(x):
            new_x_shape = concat([
                batch_size,
                -1, attn.num_attention_heads, attn.attention_head_size
            ])

            y = x.view(new_x_shape)
            y = y.permute([0, 2, 3, 1])
            return y

        query = transpose_for_scores(query)
        key = transpose_for_scores_k(key)
        value = transpose_for_scores(value)

        attention_scores = matmul(query, key, use_fp32_acc=False)
        attention_probs = softmax(attention_scores, dim=-1)

        context = matmul(attention_probs, value, use_fp32_acc=False).transpose(1, 2)
        context = context.view(
            concat([
                shape(context, 0),
                shape(context, 1), attn.attention_hidden_size
            ]))
        return attn.to_out(context)

# DiT Block
class DiTBlock(Module):

    def __init__(self, dim, heads, dim_head, ff_mult = 2, dropout = 0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor = AttnProcessor(),
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            )

        self.ff_norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = dropout, approximate = "tanh")

    def forward(self, x, t, rope_cos, rope_sin, mask = None, scale = 1.0, rope = ModuleNotFoundError): # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        # attention
        # norm ----> (2,1226,1024)
        attn_output = self.attn(x=norm, rope_cos=rope_cos, rope_sin=rope_sin, mask=mask, scale=scale)

        # process attention output for input x
        x = x + unsqueeze(gate_msa, 1) * attn_output
        ones = constant(np.ones(1, dtype = np.float32)).cast(x.dtype)
        norm = self.ff_norm(x) * (ones + unsqueeze(scale_mlp, 1)) + unsqueeze(shift_mlp, 1)
        ff_output = self.ff(norm)
        x = x + unsqueeze(gate_mlp, 1) * ff_output

        return x

class SinusPositionEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = exp(arange(start=0, end=half_dim, dtype=trt_dtype_to_str(trt.float32)) * - emb)
        emb = scale * unsqueeze(x, 1) * unsqueeze(emb, 0)
        emb = concat([cos(emb), sin(emb)], dim=-1)
        emb = emb.cast(x.dtype)
        assert self.dim % 2 == 0
        return emb

class TimestepEmbedding(Module):
    def __init__(self, dim, freq_embed_dim=256, dtype=None):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.mlp1 = Linear(freq_embed_dim, dim, bias=True, dtype=dtype)
        self.mlp2 = Linear(dim, dim, bias=True, dtype=dtype)

    def forward(self, timestep: float["b n"]):  # noqa: F821
        t_freq = self.mlp1(timestep)
        t_freq = silu(t_freq)
        t_emb = self.mlp2(t_freq)
        return t_emb

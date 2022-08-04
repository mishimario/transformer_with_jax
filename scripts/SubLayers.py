''' Define the sublayers in encoder/decoder layer '''

from Modules import ScaledDotProductAttention

import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    # def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    #     super().__init__()
    #
    #     self.n_head = n_head
    #     self.d_k = d_k
    #     self.d_v = d_v
    #
    #     self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    #     self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    #     self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    #     self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    #
    #     self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
    #
    #     self.dropout = nn.Dropout(dropout)
    #     self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    n_head: int
    d_model: int
    d_k: int
    d_v: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = nn.Dense(n_head*d_k, use_bias=False)(q).reshape((sz_b, len_q, n_head, d_k))
        k = nn.Dense(n_head*d_k, use_bias=False)(k).reshape((sz_b, len_k, n_head, d_k))
        v = nn.Dense(n_head*d_v, use_bias=False)(v).reshape((sz_b, len_v, n_head, d_v))

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose((0, 2, 1, 3)), k.transpose((0, 2, 1, 3)), v.transpose((0, 2, 1, 3))

        if mask is not None:
            mask = mask.expand_dims(1)   # For head axis broadcasting.

        q, attn = ScaledDotProductAttention(temperature=d_k ** 0.5)(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose((0, 2, 1, 3)).reshape((sz_b, len_q, -1))
        q = nn.Dropout(dropout)(nn.Dense(d_model, use_bias=False)(q))
        q += residual

        q = nn.LayerNorm(epsilon=1e-6)(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    # def __init__(self, d_in, d_hid, dropout=0.1):
    #     super().__init__()
    #     self.w_1 = nn.Linear(d_in, d_hid) # position-wise
    #     self.w_2 = nn.Linear(d_hid, d_in) # position-wise
    #     self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    #     self.dropout = nn.Dropout(dropout)
    d_out: int
    d_hid: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x):

        residual = x
        # x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)

        x = nn.Dense(self.d_hid)(x)
        x = nn.relu(x)
        x = nn.Dense(self.d_out)(x)
        x += residual

        x = nn.LayerNorm(epsilon=1e-6)(x)

        return x


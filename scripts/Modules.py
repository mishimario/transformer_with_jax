import jax
import jax.numpy as jnp
import flax.linen as nn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    # def __init__(self, temperature, attn_dropout=0.1):
    #     super().__init__()
    #     self.temperature = temperature
    #     self.dropout = nn.Dropout(attn_dropout)
    temperature: float
    attn_dropout: float = 0.1

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        attn = jnp.matmul(q / self.temperature, k.transpose((0, 1, 3, 2)))

        if mask is not None:
            jnp.where(mask, -1e9, attn)
            # attn = attn.masked_fill(mask == 0, -1e9)

        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = nn.Dropout(self.attn_dropout)(nn.softmax(attn, dim=-1))
        output = jnp.matmul(attn, v)

        return output, attn

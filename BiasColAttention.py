import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasColumnSelfAttention(nn.Module):
    def __init__(
        self,
        d: int,
        L: int,
        heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        col_bias: bool = True,
    ):
        super().__init__()
        assert d % heads == 0, (
            f"d_model={d} 必须能被 heads={heads} 整除"
        )

        self.d = d
        self.L = L
        self.heads = heads
        self.dropout = dropout
        self.col_bias = col_bias

        # 四组线性变换，均带偏置
        self.W_q = nn.Linear(d, d, bias=bias)
        self.W_k = nn.Linear(d, d, bias=bias)
        self.W_v = nn.Linear(d, d, bias=bias)
        self.W_o = nn.Linear(d, d, bias=bias)
        if self.col_bias:
            self.bias = nn.Parameter(torch.zeros(L)) #为每列提供偏置

    def forward(self, x: torch.Tensor,
                heads: int | None = None,
                padding_mask = None,
                return_attn = False):

        if heads is None:
            heads = self.heads
        assert heads > 0, f"heads 必须为正整数，实际为 {heads}"
        assert self.d % heads == 0, (
            f"d_model={self.d} 必须能被 heads={heads} 整除"
        )

        B, N, L, D = x.shape
        assert D == self.d, (
            f"输入最后一维 {D} 有误"
        )
        assert L == self.L, (
            f"输入序列长度 {L} 有误"
        )

        d_head = D // heads

        x_flat = x.permute(0, 2, 1, 3).reshape(B * L, N, D)

        q = self.W_q(x_flat)                             # [B*L, N, d]
        k = self.W_k(x_flat)                             # [B*L, N, d]
        v = self.W_v(x_flat)                             # [B*L, N, d]

        q = q.reshape(B * L, N, heads, d_head).transpose(1, 2)
        k = k.reshape(B * L, N, heads, d_head).transpose(1, 2)
        v = v.reshape(B * L, N, heads, d_head).transpose(1, 2)

        A = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d_head))

        if padding_mask is not None:
            assert torch.is_tensor(padding_mask) and padding_mask.shape == (B, N, L), f"padding_mask格式有误"
            mask_flat = padding_mask.permute(0,2,1).reshape(B * L, N)
            A = A.masked_fill(mask_flat[:, None, None, :], torch.finfo(A.dtype).min)

        A = F.softmax(A, dim=-1)

        if self.col_bias:
            bias_flat = self.bias.unsqueeze(0).expand(B, L).reshape(B * L)  # [B*L]
            A = A + bias_flat.view(B * L, 1, 1, 1)

        if return_attn:
            attn = A.mean(dim=1)  # [B*L, N, N]
            attn = attn.reshape(B, L, N, N).contiguous() #返回全部注意力图
        else:
            attn = None

        A = torch.nan_to_num(A, nan=0.0)
        A = F.dropout(A, p=self.dropout, training=self.training)

        out = torch.matmul(A, v)
        out = out.transpose(1, 2).reshape(B * L, N, D)
        out = self.W_o(out)
        out = out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        if return_attn:
            return out.contiguous(), attn
        return out.contiguous()
import torch
import torch.nn as nn
from RowAttention import RowSelfAttention
from BiasColAttention import BiasColumnSelfAttention   # 按实际模块名调整


class BiasTransformerBlock(nn.Module):
    def __init__(
        self,
        L: int, #需要传入 L
        d: int = 768,
        heads: int = 12,
        d_ffn: int = 3072,
        row_dropout: float = 0.0,
        column_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        bias: bool = True,
        col_bias: bool = True,
    ):
        super().__init__()
        assert d % heads == 0, f"d={d} 必须能被 heads={heads} 整除"
        assert d_ffn > 0, f"d_ffn 必须为正整数，实际为 {d_ffn}"
        assert L > 0, f"L 必须为正整数，实际为 {L}"

        self.d = d
        self.L = L
        self.heads = heads
        self.d_ffn = d_ffn

        self.row_attn = RowSelfAttention(
            d=d, heads=heads, dropout=row_dropout, bias=bias
        )
        self.col_attn = BiasColumnSelfAttention(
            d=d, L=L, heads=heads, dropout=column_dropout, bias=bias, col_bias=col_bias
        )

        self.ln_row = nn.LayerNorm(d)
        self.ln_col = nn.LayerNorm(d)
        self.ln_ffn = nn.LayerNorm(d)

        self.ffn_linear1 = nn.Linear(d, d_ffn, bias=bias)
        self.ffn_linear2 = nn.Linear(d_ffn, d, bias=bias)
        self.ffn_act = nn.GELU()
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, L, d] -> [B, N, L, d]"""
        x = self.ffn_linear1(x)
        x = self.ffn_act(x)
        x = self.ffn_dropout(x)
        x = self.ffn_linear2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        padding_mask = None,
        heads: int | None = None,
        return_row_attn: bool = False,
        return_col_attn: bool = False,):

        B, N, L, D = x.shape
        assert D == self.d, f"输入最后一维 {D} 与 d={self.d} 不一致"
        assert L == self.L, f"输入 L={L} 与初始化时的 self.L={self.L} 不一致"

        h = self.ln_row(x)
        if return_row_attn:
            h, row_attn = self.row_attn(h, heads=heads, padding_mask=padding_mask, return_attn=True)
        else:
            h, row_attn = self.row_attn(h, heads=heads, padding_mask=padding_mask), None
        x = x + h

        h = self.ln_col(x)
        if return_col_attn:
            h, col_attn = self.col_attn(h, heads=heads, padding_mask=padding_mask, return_attn=True)
        else:
            h, col_attn = self.col_attn(h, heads=heads, padding_mask=padding_mask), None
        x = x + h

        h = self.ln_ffn(x)
        h = self.ffn(h)
        x = x + h

        if return_row_attn and return_col_attn:
            return x, row_attn, col_attn
        elif return_row_attn:
            return x, row_attn
        elif return_col_attn:
            return x, col_attn
        return x
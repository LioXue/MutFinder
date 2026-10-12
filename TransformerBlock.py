import torch
import torch.nn as nn
from RowAttention import RowSelfAttention
from ColumnAttention import ColumnSelfAttention

class MSATransformerBlock(nn.Module):
    def __init__(
        self,
        d: int = 768,
        heads: int = 12,
        d_ffn: int = 3072,
        row_dropout: float = 0.0,
        column_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d % heads == 0, f"d={d} 必须能被 heads={heads} 整除"
        assert d_ffn > 0, f"d_ffn 必须为正整数，实际为 {d_ffn}"

        self.d = d
        self.heads = heads
        self.d_ffn = d_ffn

        self.row_attn = RowSelfAttention(d=d, heads=heads, dropout=row_dropout, bias=bias)
        self.col_attn = ColumnSelfAttention(d=d, heads=heads, dropout=column_dropout, bias=bias)

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
        heads: int | None = None,) -> torch.Tensor:

        B, N, L, D = x.shape
        assert D == self.d, f"输入最后一维 {D} 与 d={self.d} 不一致"

        h = self.ln_row(x)
        h = self.row_attn(h, heads=heads, padding_mask=padding_mask)
        x = x + h

        h = self.ln_col(x)
        h = self.col_attn(h, heads=heads, padding_mask=padding_mask)
        x = x + h

        h = self.ln_ffn(x)
        h = self.ffn(h)
        x = x + h

        return x
import torch
import torch.nn as nn

from Embedding import ESMMSA1bEmbedding
from TransformerBlock import MSATransformerBlock
from BiasTransBlock import BiasTransformerBlock


class MutFinder(nn.Module):
    def __init__(
        self,
        embed_layer: ESMMSA1bEmbedding,
        L: int,                          # 需要传入序列长度
        d: int = 768,
        heads: int = 12,
        d_ffn: int = 3072,
        num_msa_blocks: int = 5,
        row_dropout: float = 0.0,
        column_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        col_bias: bool = True,
        return_attns: bool = False,
    ):
        super().__init__()

        self.L = L
        self.d = d
        self.embed_layer = embed_layer
        self.alphabet = embed_layer.alphabet
        self.padding_idx = embed_layer.padding_idx
        self.return_attns = return_attns

        self.msa_blocks = nn.ModuleList([
            MSATransformerBlock(
                d=d, heads=heads, d_ffn=d_ffn,
                row_dropout=row_dropout,
                column_dropout=column_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(num_msa_blocks)
        ])

        self.bias_block = BiasTransformerBlock(
            L=L, d=d, heads=heads, d_ffn=d_ffn,
            row_dropout=row_dropout,
            column_dropout=column_dropout,
            ffn_dropout=ffn_dropout,
            col_bias=col_bias,
        )

        self.head = nn.Linear(L, 1, bias=True)

    def forward(self, tokens: torch.Tensor):
        assert torch.is_tensor(tokens), "tokens 必须是 torch.Tensor"
        assert tokens.dtype == torch.long, \
            f"tokens dtype 必须为 torch.long，实际为 {tokens.dtype}"
        assert tokens.dim() == 3, \
            f"tokens 形状必须为 [B, N, L]，实际为 {tuple(tokens.shape)}"

        B, N, L = tokens.shape
        assert L == self.L, \
            f"输入 L={L} 与模型 self.L={self.L} 不一致"

        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_layer(tokens)

        for i, block in enumerate(self.msa_blocks):
            x = block(x, padding_mask=padding_mask)

        if self.return_attns:
            x, row_attn, col_attn = self.bias_block(x, padding_mask=padding_mask,
                                                    return_row_attn=self.return_attns,
                                                    return_col_attn=self.return_attns)
        else:
            x, row_attn, col_attn = self.bias_block(x, padding_mask=padding_mask), None, None

        x = x.mean(dim=-1)
        logits = self.head(x).squeeze(-1)

        return logits, x, row_attn, col_attn
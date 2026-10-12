import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnSelfAttention(nn.Module):
    """
    MSA Transformer 列注意力子层（多头自注意力）。

    对输入 MSA 张量 [B, N, L, d] 沿 **N 维** 做自注意力：
    - 先转置为 [B, L, N, d]，再展平为 [B*L, N, d]
    - 每个 (batch, 位置) 被当作一个独立的 (N, d) 矩阵
    - B*L 个矩阵**共享** W_q / W_k / W_v / W_o 参数
    - Q = W_q(X), K = W_k(X), V = W_v(X)
    - A = softmax(Q @ K^T / sqrt(d_head))
    - A = dropout(A)
    - output = W_o(A @ V)

    输入
    ----
    x : torch.Tensor, 形状 [B, N, L, d]

    输出
    ----
    out : torch.Tensor, 形状 [B, N, L, d]（形状与输入完全一致）

    参数
    ----
    d        : int   特征维度 d（即输入最后一维）
    heads    : int   默认注意力头数，必须整除 d
    dropout  : float 注意力图 dropout 概率
    bias     : bool  线性层是否带偏置
    """

    def __init__(
        self,
        d: int,
        heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d % heads == 0, (
            f"d_model={d} 必须能被 heads={heads} 整除"
        )

        self.d = d
        self.heads = heads
        self.dropout = dropout

        # 四组线性变换，均带偏置
        self.W_q = nn.Linear(d, d, bias=bias)
        self.W_k = nn.Linear(d, d, bias=bias)
        self.W_v = nn.Linear(d, d, bias=bias)
        self.W_o = nn.Linear(d, d, bias=bias)

    def forward(self, x: torch.Tensor,
                heads: int | None = None,
                padding_mask = None) -> torch.Tensor:
        """
        x     : [B, N, L, d]
        heads : 可选，覆盖 __init__ 时的默认头数（必须整除 d）
        """
        if heads is None:
            heads = self.heads
        assert heads > 0, f"heads 必须为正整数，实际为 {heads}"
        assert self.d % heads == 0, (
            f"d_model={self.d} 必须能被 heads={heads} 整除"
        )

        B, N, L, D = x.shape
        assert D == self.d, (
            f"输入最后一维 {D} 与 d={self.d} 不一致"
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
        A = torch.nan_to_num(A, nan=0.0)
        A = F.dropout(A, p=self.dropout, training=self.training)

        out = torch.matmul(A, v)
        out = out.transpose(1, 2).reshape(B * L, N, D)
        out = self.W_o(out)
        out = out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        return out.contiguous()
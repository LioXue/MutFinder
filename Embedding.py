import torch
import torch.nn as nn
import esm
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

class ESMMSA1bEmbedding(nn.Module):
    """
    基于 ESM-MSA-1b 的第 0 层嵌入层。
    - ESM-MSA-1b 全部参数冻结，模块始终处于 eval() 模式
    - 不构建 Transformer 主干，只暴露 embed_tokens / embed_positions 进行 token编码和列位置编码

    输入
    将 list(str):[seq1, seq2, ...] 通过 alphabet encode成 list(list(int)):[[2,3,...],...]
    最后转化为 torch.long张量
    *** tokens: LongTensor(torch.long), 形状 [B, N, L]
            B = batch size
            N = MSA 深度（同源序列条数）
            L = 序列长度（对齐后的列数）

    输出
    embeddings: FloatTensor, 形状 [B, N, L, D]，D = 768
    """

    def __init__(
        self,
        include_token: bool = True, #token嵌入
        include_position: bool = True, #列位置嵌入
        apply_layer_norm: bool = False, #layer正则化
        zero_padding: bool = True, #padding位置清零
    ):
        super().__init__()
        model, alphabet = esm.pretrained.load_model_and_alphabet_local('./model_weight/esm_msa1b_t12_100M_UR50S.pt')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.esm = model
        self.alphabet = alphabet

        self.embed_dim   = int(model.args.embed_dim)   # 768
        self.embed_scale = float(self.embed_dim ** 0.5)
        self.padding_idx = int(alphabet.padding_idx)
        self._has_layer_norm = getattr(model, "emb_layer_norm_before", None) is not None

        self.include_token    = bool(include_token)
        self.include_position = bool(include_position)
        self.apply_layer_norm = bool(apply_layer_norm and self._has_layer_norm)
        self.zero_padding     = bool(zero_padding)

        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "ESMMSA1bEmbedding":
        super().train(False)
        self.esm.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.esm.embed_tokens.weight.device

    def _check_input(self, tokens: torch.Tensor) -> None:
        if not torch.is_tensor(tokens):
            raise TypeError("tokens 必须是 torch.Tensor")
        if tokens.dim() != 3:
            raise ValueError(
                f"tokens 形状必须为 [B, N, L]，实际为 {tuple(tokens.shape)}"
            )
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens dtype 必须为 torch.long，实际为 {tokens.dtype}")

    def token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """仅返回 token 身份嵌入（含 embed_scale），形状 [B, N, L, D]。"""
        self._check_input(tokens)
        return self.embed_scale * self.esm.embed_tokens(tokens)

    def position_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """仅返回列位置嵌入（每行独立按列索引取），形状 [B, N, L, D]。"""
        self._check_input(tokens)
        B, N, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B * N, L)
        pos = self.esm.embed_positions(positions)
        return pos.reshape(B, N, L, self.embed_dim)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self._check_input(tokens)
        B, N, L = tokens.shape
        D = self.embed_dim

        x = torch.zeros(
            (B, N, L, D),
            dtype=self.esm.embed_tokens.weight.dtype,
            device=tokens.device,
        )

        # token 编码
        if self.include_token:
            x = x + self.token_embedding(tokens)

        # column position 编码
        if self.include_position:
            x = x + self.position_embedding(tokens)

        if self.apply_layer_norm:
            x = self.esm.emb_layer_norm_before(x)

        if self.zero_padding:
            pad_mask = tokens.eq(self.padding_idx)
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return x
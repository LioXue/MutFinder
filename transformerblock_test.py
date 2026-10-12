"""
test_msa_block.py
用 [1, 3, 200, 768] 的随机输入 + padding_mask 检验 MSATransformerBlock。
"""
import torch

from TransformerBlock import MSATransformerBlock   # 按实际模块名调整


def banner(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- 1) 输入配置 ----
    B, N, L, d = 1, 3, 200, 768
    HEADS = 12
    D_FFN = 3072

    banner("[1] 构造输入")
    x = torch.randn(B, N, L, d, device=device)
    print(f"  x.shape = {tuple(x.shape)}  dtype={x.dtype}")

    # padding_mask: [B, N, L] bool, True = padding
    padding_mask = torch.zeros(B, N, L, dtype=torch.bool, device=device)
    padding_mask[0, 2, 150:] = True     # 第 2 行（第 3 条同源序列）后 50 个位置是 padding
    print(f"  padding_mask.shape = {tuple(padding_mask.shape)}  "
          f"dtype={padding_mask.dtype}")
    print(f"  padding 数量 = {padding_mask.sum().item()} / {padding_mask.numel()}")

    # ---- 2) 构造 Block ----
    banner("[2] 构造 MSATransformerBlock")
    block = MSATransformerBlock(
        d=d, heads=HEADS, d_ffn=D_FFN,
        row_dropout=0.0, column_dropout=0.0, ffn_dropout=0.0,
        bias=True,
    ).to(device).eval()

    n_params = sum(p.numel() for p in block.parameters())
    print(f"  参数量 = {n_params:,}  (≈ {n_params/1e6:.2f} M)")
    print(f"  d={block.d}, heads={block.heads}, d_ffn={block.d_ffn}")

    # ---- 3) 前向：带 mask ----
    banner("[3] 前向（带 padding_mask）")
    with torch.no_grad():
        y = block(x, padding_mask=padding_mask)
    print(f"  y.shape = {tuple(y.shape)}")
    assert y.shape == x.shape, f"输出形状 {y.shape} != 输入 {x.shape}"
    assert torch.isfinite(y).all(), "存在 NaN/Inf"
    print(f"  输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("  [PASS] 形状一致、无 NaN/Inf")

    # ---- 4) 前向：不带 mask ----
    banner("[4] 前向（不带 mask）")
    with torch.no_grad():
        y_no_mask = block(x)
    assert y_no_mask.shape == x.shape
    assert torch.isfinite(y_no_mask).all()
    diff = (y - y_no_mask).abs().max().item()
    print(f"  y_no_mask.shape = {tuple(y_no_mask.shape)}")
    print(f"  与带 mask 输出的 max_diff = {diff:.4e}  (应 > 0，mask 确实起作用)")
    assert diff > 0, "带 mask 与不带 mask 输出完全相同，mask 可能没生效"
    print("  [PASS] mask 确实改变了输出")

    # ---- 5) mask 生效验证：扰动 padding 位置应不影响有效位置 ----
    banner("[5] mask 有效性验证（扰动实验）")
    x_perturbed = x.clone()
    # 只扰动 padding 位置（n=2, l>=150）的输入
    x_perturbed[0, 2, 150:, :] = torch.randn_like(x_perturbed[0, 2, 150:, :]) * 10.0

    with torch.no_grad():
        y_perturbed = block(x_perturbed, padding_mask=padding_mask)

    # 有效位置的输出应该几乎不变（除行/列注意力本身的 padding 行 query 输出）
    # 说明：padding 位置作为 query 的输出本来就会变；作为 key 被 mask，不应影响其他位置
    diff_valid = (y[0, 0] - y_perturbed[0, 0]).abs().max().item()   # 第 0 行完全有效
    print(f"  第 0 行（完全有效）的输出差异 = {diff_valid:.3e}")

    # 由于整个 block 有 FFN 和残差，允许很小误差
    assert diff_valid < 1e-4, \
        f"padding key 扰动了有效 query 的输出，diff={diff_valid}"
    print("  [PASS] padding key 不影响有效 query 的输出")

    # ---- 6) 梯度回传 ----
    banner("[6] 梯度回传")
    x_g = x.clone().requires_grad_(True)
    y_g = block(x_g, padding_mask=padding_mask)
    loss = y_g.pow(2).mean()
    loss.backward()
    assert x_g.grad is not None and torch.isfinite(x_g.grad).all()
    print(f"  loss = {loss.item():.4f}")
    print(f"  x.grad 范围: [{x_g.grad.min().item():.3e}, {x_g.grad.max().item():.3e}]")
    print("  [PASS] 梯度回传正常")

    # ---- 7) 参数更新 ----
    banner("[7] 优化器一步")
    block2 = MSATransformerBlock(d=d, heads=HEADS, d_ffn=D_FFN).to(device)
    opt = torch.optim.SGD(block2.parameters(), lr=1e-3)
    w_before = block2.ffn_linear1.weight.detach().clone()
    y2 = block2(x, padding_mask=padding_mask)
    y2.pow(2).mean().backward()
    opt.step()
    diff_w = (block2.ffn_linear1.weight.detach() - w_before).abs().max().item()
    assert diff_w > 0
    print(f"  ffn_linear1.weight 变化 max = {diff_w:.3e}")
    print("  [PASS] 参数可以被优化器更新")

    # ---- 8) state_dict 检查 ----
    banner("[8] state_dict 键检查")
    keys = list(block.state_dict().keys())
    print(f"  state_dict 共 {len(keys)} 个键")
    has_gelu_param = any("ffn_act" in k for k in keys)
    print(f"  含 ffn_act 的键: {has_gelu_param}  (GELU 无参数，应为 False)")
    # 抽查关键子模块键
    for name in ("row_attn.W_q.weight", "col_attn.W_q.weight",
                 "ln_row.weight", "ffn_linear1.weight", "ffn_linear2.weight"):
        assert name in keys, f"缺少键: {name}"
    print("  [PASS] 所有子模块已正确注册")

    banner("全部测试通过 ✅")


if __name__ == "__main__":
    main()
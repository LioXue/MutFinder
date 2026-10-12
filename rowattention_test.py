"""
row_attention_test.py
检验 RowSelfAttention 在输入 [B=2, N=38, L=218, d=768] 下能否正常运行。
"""
import math
import torch
import torch.nn.functional as F

from RowAttention import RowSelfAttention   # 按实际模块名调整


# --------------------------------------------------------------------------
# 固定测试配置
# --------------------------------------------------------------------------
B, N, L, D = 2, 38, 218, 768
HEADS = 12          # 768 / 12 = 64，合法
DROPOUT = 0.1


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# --------------------------------------------------------------------------
# 1. 形状与数值稳定性
# --------------------------------------------------------------------------
def test_shape_and_finite(layer, device):
    banner("[1] 形状 & 数值稳定性")
    x = torch.randn(B, N, L, D, device=device)
    print(f"  x.shape = {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")

    layer.eval()
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape, f"输出 {tuple(y.shape)} != 输入 {tuple(x.shape)}"
    assert torch.isfinite(y).all(), "存在 NaN/Inf"

    print(f"  y.shape = {tuple(y.shape)}")
    print(f"  输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("  [PASS] 形状一致、无 NaN/Inf")


# --------------------------------------------------------------------------
# 2. 梯度回传
# --------------------------------------------------------------------------
def test_backward(layer, device):
    banner("[2] 梯度回传")
    x = torch.randn(B, N, L, D, device=device, requires_grad=True)

    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("W_q", "W_k", "W_v", "W_o"):
        w = getattr(layer, name).weight
        assert w.grad is not None and torch.isfinite(w.grad).all(), f"{name} 梯度异常"

    print(f"  loss = {loss.item():.4f}")
    print(f"  x.grad 范围: [{x.grad.min().item():.3e}, {x.grad.max().item():.3e}]")
    print("  [PASS] 所有参数与输入均收到有限梯度")


# --------------------------------------------------------------------------
# 3. 多头接口（合法 / 非法 head 数）
# --------------------------------------------------------------------------
def test_heads(layer, device):
    banner("[3] 多头接口")
    x = torch.randn(B, N, L, D, device=device)
    layer.eval()

    for h in (1, 2, 4, 8, 12, 16, 24, 32):
        if D % h != 0:
            continue
        with torch.no_grad():
            y = layer(x, heads=h)
        assert y.shape == x.shape
        print(f"  heads={h:2d}  d_head={D//h:3d}  y.shape={tuple(y.shape)}  [OK]")

    # 非法头数
    try:
        with torch.no_grad():
            layer(x, heads=5)     # 768 % 5 != 0
        raise AssertionError("heads=5 应报错但没报")
    except AssertionError as e:
        if "应报错" in str(e):
            raise
        print("  heads=5   →  AssertionError  [符合预期]")

    print("  [PASS] 多头接口正常")


# --------------------------------------------------------------------------
# 4. dropout 行为
# --------------------------------------------------------------------------
def test_dropout(device):
    banner("[4] dropout 行为")
    layer = RowSelfAttention(d=D, heads=HEADS, dropout=0.5, bias=True).to(device)
    x = torch.randn(B, N, L, D, device=device)

    layer.eval()
    with torch.no_grad():
        y1 = layer(x); y2 = layer(x)
    assert (y1 - y2).abs().max().item() == 0
    print("  eval  模式两次输出完全一致  [OK]")

    layer.train()
    with torch.no_grad():
        y3 = layer(x); y4 = layer(x)
    d_train = (y3 - y4).abs().max().item()
    assert d_train > 0, "train 模式输出相同，dropout 未生效"
    print(f"  train 模式两次输出 max_diff = {d_train:.3e}  [OK]")
    print("  [PASS] dropout 行为正确")


# --------------------------------------------------------------------------
# 5. 注意力图行和为 1
# --------------------------------------------------------------------------
def test_softmax(layer, device):
    banner("[5] 注意力图行和为 1")
    x = torch.randn(B, N, L, D, device=device)
    layer.eval()

    with torch.no_grad():
        x_flat = x.reshape(B * N, L, D)
        d_head = D // layer.heads
        q = layer.W_q(x_flat).reshape(B * N, L, layer.heads, d_head).transpose(1, 2)
        k = layer.W_k(x_flat).reshape(B * N, L, layer.heads, d_head).transpose(1, 2)
        A = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        A = F.softmax(A, dim=-1)

    row_sums = A.sum(dim=-1)                         # [B*N, heads, L]
    diff = (row_sums - 1.0).abs().max().item()
    assert diff < 1e-5, f"行和不为 1，max_diff={diff}"
    print(f"  A.shape = {tuple(A.shape)}   (期望 [B*N={B*N}, heads={layer.heads}, L={L}, L={L}])")
    print(f"  行和 max_diff = {diff:.3e}")
    print("  [PASS] 注意力图每行和为 1")


# --------------------------------------------------------------------------
# 6. 显存 & 耗时（可选，参考用）
# --------------------------------------------------------------------------
def test_perf(layer, device):
    banner("[6] 性能参考")
    x = torch.randn(B, N, L, D, device=device)
    layer.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        with torch.no_grad():
            _ = layer(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"  单次前向耗时 ≈ {dt*1000:.1f} ms")
        print(f"  峰值显存 ≈ {mem:.1f} MB")
    else:
        import time
        t0 = time.time()
        with torch.no_grad():
            _ = layer(x)
        dt = time.time() - t0
        print(f"  CPU 单次前向耗时 ≈ {dt*1000:.1f} ms")
    print("  [INFO] 性能数据仅供观察，不作断言")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"输入形状: B={B}, N={N}, L={L}, d={D}")
    print(f"heads={HEADS}, d_head={D // HEADS}")

    # 主实例（无 dropout，便于确定性测试）
    layer = RowSelfAttention(d=D, heads=HEADS, dropout=0.0, bias=True).to(device)

    test_shape_and_finite(layer, device)
    test_backward(layer, device)
    test_heads(layer, device)
    test_dropout(device)               # 内部自建 dropout 实例
    test_softmax(layer, device)
    test_perf(layer, device)           # 仅参考

    print("\n" + "=" * 64)
    print("全部测试通过 ✅")
    print("=" * 64)


if __name__ == "__main__":
    main()
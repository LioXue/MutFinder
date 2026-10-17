import csv
import torch
from Embedding import ESMMSA1bEmbedding
from MutFinder import MutFinder


def load_pred_csv(csv_path):
    seqs_0, seqs_1 = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (seqs_0 if int(row["label"]) == 0 else seqs_1).append(row["sequence"])
    return seqs_0, seqs_1


def encode_msa(seqs, alphabet, padding_idx, L, device):
    N = len(seqs)
    tokens = torch.full((1, N, L), padding_idx, dtype=torch.long, device=device)
    for n, seq in enumerate(seqs):
        ids = alphabet.encode(seq)[:L]
        tokens[0, n, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return tokens


def predict(csv_path,
            model_path,
            maxtop=10,
            attntop=10,
            heads=12,
            d_ffn=3072,
            num_msa_blocks=5,
            col_bias=True,
            device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weight = torch.load(model_path, map_location=device)
    max_L = model_weight["max_L"]
    print(f"[predict] 加载模型{model_path}，max_L = {max_L}")

    embed_layer = ESMMSA1bEmbedding()
    model = MutFinder(embed_layer=embed_layer,
                      L=max_L,
                      heads=heads,
                      d_ffn=d_ffn,
                      num_msa_blocks=num_msa_blocks, #与train主程序定义一致
                      col_bias=col_bias,
                      return_attns=True,
                      ).to(device)
    model.load_state_dict(model_weight["model_state"])
    model.eval()

    alphabet = embed_layer.alphabet
    padding_idx = embed_layer.padding_idx

    seqs_0, seqs_1 = load_pred_csv(csv_path)
    print(f"[predict] label=0: {len(seqs_0)} 条，label=1: {len(seqs_1)} 条")
    assert len(seqs_0) > 0, "没有 label=0 的序列"
    assert len(seqs_1) > 0, "没有 label=1 的序列"

    tokens_0 = encode_msa(seqs_0, alphabet, padding_idx, max_L, device)  # [1, A, L]
    tokens_1 = encode_msa(seqs_1, alphabet, padding_idx, max_L, device)  # [1, B, L]
    print(f"[predict] tokens_0 = {tuple(tokens_0.shape)}, "
          f"tokens_1 = {tuple(tokens_1.shape)}")

    with torch.no_grad():
        _, x_0, row_attn0, col_attn0 = model(tokens_0)   # x_0: [1, A, L], row_attn: [1,N,L,L]
        _, x_1, row_attn1, col_attn1 = model(tokens_1)   # x_1: [1, B, L], col_attn: [1,L,N,N]

    a = x_0.mean(dim=1).squeeze(0)     # [L]
    b = x_1.mean(dim=1).squeeze(0)     # [L]

    V = model.head.weight.squeeze(0)   # [L]

    aV = a * V                          # [L]
    bV = b * V                          # [L]
    diff = aV - bV

    max_values, max_indices = torch.topk(diff.abs(), k=min(maxtop, max_L))
    for i, (v, idx) in enumerate(zip(max_values.tolist(), max_indices.tolist()), 1):
        print(f"#{i}: 位置{idx:>4d}  value = {v:+.6f}")

    ar = row_attn0.mean(dim=1).squeeze(0)
    br = row_attn1.mean(dim=1).squeeze(0)
    r_diff = ar - br
    row_max_values, row_max_indices = torch.topk(r_diff.flatten(), k=min(attntop, max_L**2))
    for i, (v, idx) in enumerate(zip(row_max_values.tolist(), row_max_indices.tolist()), 1):
        idx1 = idx // r_diff.size(1)
        idx2= idx % r_diff.size(1)
        print(f"#{i}: 互作{idx1}-{idx2}  value = {v:+.6f}")
    row_min_values, row_min_indices = torch.topk(r_diff.flatten(), k=min(attntop, max_L**2), largest=False)
    for i, (v, idx) in enumerate(zip(row_min_values.tolist(), row_min_indices.tolist()), 1):
        idx1 = idx // r_diff.size(1)
        idx2= idx % r_diff.size(1)
        print(f"#{i}: 互作{idx1}-{idx2}  value = {v:+.6f}")

    #ac = col_attn0.mean(dim=1).squeeze(0)
    #bc = col_attn1.mean(dim=1).squeeze(0)
    #c_diff = ac - bc
    #col_max_values, col_max_indices = torch.topk(c_diff.flatten(), k=min(maxtop, c_diff.shape[0]*c_diff.shape[1]))
    #for i, (v, idx) in enumerate(zip(col_max_values.tolist(), col_max_indices.tolist()), 1):
    #    idx1 = idx // c_diff.size(1)
    #    idx2= idx % c_diff.size(1)
    #    print(f"#{i}: 互作{idx1}-{idx2}  value = {v:+.6f}")
    #col_min_values, col_min_indices = torch.topk(c_diff.flatten(), k=min(maxtop, c_diff.shape[0]*c_diff.shape[1]), largest=False)
    #for i, (v, idx) in enumerate(zip(col_min_values.tolist(), col_min_indices.tolist()), 1):
    #    idx1 = idx // c_diff.size(1)
    #    idx2 = idx % c_diff.size(1)
    #    print(f"#{i}: 互作{idx1}-{idx2}  value = {v:+.6f}")

    aux = {
        "a": a.cpu(),
        "b": b.cpu(),
        "V": V.cpu(),
        "aV": aV.cpu(),
        "bV": bV.cpu(),
        "diff": diff.cpu(),
    }
    return aux

if __name__ == "__main__":
    max_indices, max_values, min_values, min_indices, aux = predict(
        csv_path="./data/50a_0.3_0_1.0_grande_msa.csv",
        model_path="./model_weight/50a_0.3_0_1.0_mutfinder.pt",

        heads=12,
        d_ffn=3072,
        num_msa_blocks=0,
        col_bias=True,

        maxtop=10,
        attntop=10,
    )
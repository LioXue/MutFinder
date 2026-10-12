import torch
import torch.nn.functional as F
import os

from embedding import ESMMSA1bEmbedding
from MutFinder import MutFinder
from Data import build_dataloader


def train(
    csv_path: str = "./data/random_msa.csv",
    model_path: str = "./model_weight/mutfinder.pt",
    epoch_model: int = 0,
    heads = 12,
    d_ffn = 3072,
    num_msa_blocks = 5,
    row_dropout = 0.0,
    column_dropout = 0.0,
    ffn_dropout = 0.0,
    col_bias = True,
    num_epochs: int = 10,
    early_stop: int = 5,
    lr: float = 1e-4,
    decay_rate: float = 0,
    l :str = "Null",
    n_msa: int = 20,  # 生成的msa数量
    n_seq: int = 25,  # 每个msa的seq数量
    batch_size: int = 4, #将msa按batch_size划分
    #seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    print(f"正在训练：{model_path}")
    print(f"基于数据：{csv_path}")

    embed_layer = ESMMSA1bEmbedding()
    alphabet = embed_layer.alphabet
    padding_idx = embed_layer.padding_idx

    loader, max_L = build_dataloader(
        csv_path,
        alphabet,
        padding_idx,
        batch_size=batch_size,
        n_msa=n_msa,
        n_seq=n_seq,
        #seed=seed,
    )
    print(f"[train] {len(loader.dataset)} 个 MSA/epoch, "
          f"{len(loader)} 个 batch, L = {max_L}")

    model = MutFinder(embed_layer=embed_layer,
                      L=max_L,
                      heads = heads,
                      d = 768,
                      d_ffn = d_ffn,
                      num_msa_blocks = num_msa_blocks,
                      row_dropout = row_dropout,
                      column_dropout = column_dropout,
                      ffn_dropout = ffn_dropout,
                      col_bias = col_bias,
                      ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    losses = []
    losses_l = []
    best_loss = float("inf")
    early_counter = 0

    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()
        epoch_loss,epoch_loss_l, n_samples = 0.0, 0.0, 0

        toprint = 0
        for tokens, labels in loader:
            print("trying", toprint)
            toprint += 1
            tokens = tokens.to(device)      # [B, N, L] long
            labels = labels.to(device)      # [B, N] float

            optimizer.zero_grad()

            logits,_,_,_ = model(tokens)          # [B, N]
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            if l == "l1":
                head_l = model.head.weight.abs().sum()
                loss_l = loss + decay_rate * head_l
            elif l == "l2":
                head_l = (model.head.weight ** 2).sum()
                loss_l = loss + decay_rate * head_l
            else:
                loss_l = loss

            loss_l.backward()

            optimizer.step()

            epoch_loss += loss.item() * tokens.size(0) * tokens.size(1)
            epoch_loss_l += loss_l.item() * tokens.size(0) * tokens.size(1)

            n_samples += tokens.size(0) * tokens.size(1)

        avg_loss = epoch_loss / n_samples
        avg_loss_l = epoch_loss_l / n_samples
        losses.append(avg_loss)
        losses_l.append(avg_loss_l)

        if verbose:
            print(f"loss = {avg_loss:.6f}\n"
                  f"loss_l2 = {avg_loss_l:.6f}")

        if epoch_model and (epoch+1) % (num_epochs//epoch_model) == 0:
            path, ext = os.path.splitext(model_path)
            new_path = f"{path}{epoch}{ext}"
            torch.save(model.state_dict(), new_path)

        if avg_loss + avg_loss_l < best_loss:
            early_counter = 0
            print("新最佳"+"*"*20)
            best_loss = avg_loss + avg_loss_l
            torch.save({
                "model_state": model.state_dict(),
                "max_L": max_L,
                "best_loss": best_loss,
            }, model_path)
        else:
            early_counter += 1

        if early_counter >= early_stop:
            break

    return model, losses


if __name__ == "__main__":
    train(
        csv_path="./data/50a_0.3_0_1.0_grande_msa.csv",
        model_path="./model_weight/50a_0.3_0_1.0_mutfinder.pt",
        epoch_model=0,

        heads=12,
        d_ffn=3072,
        num_msa_blocks=0,
        col_bias=True,

        row_dropout=0.0,
        column_dropout=0.0,
        ffn_dropout=0.0,

        num_epochs=1000,
        early_stop=30,
        #stop_value=1e-6,
        lr=1e-4,
        decay_rate=1e-2,
        l="l2",
        n_msa=2,
        n_seq=25,
        batch_size=2,
    )
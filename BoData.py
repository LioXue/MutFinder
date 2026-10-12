# data_utils.py
import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader


def load_csv(path):
    seqs, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seqs.append(row["sequence"])
            labels.append(int(row["label"]))
    return seqs, labels


class BootstrapMSADataset(Dataset):
    def __init__(self, csv_path, n_msa=20, n_seq=25, seed=42):
        self.sequences, self.labels = load_csv(csv_path)
        self.n_msa = n_msa
        self.n_seq = n_seq
        self.rng = random.Random(seed)
        self.seed = seed
        print(f"[data] 共 {len(self.sequences)} 条序列, "
              f"每 epoch {n_msa} 个 MSA, 每个 {n_seq} 条")

    def __len__(self):
        return self.n_msa

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            rng = self.rng
        else:
            rng = random.Random(self.seed + worker_info.id)
        idxs = rng.sample(range(len(self.sequences)), self.n_seq)
        msa = [self.sequences[i] for i in idxs]
        lab = [self.labels[i] for i in idxs]
        return msa, lab


def make_collate(alphabet, padding_idx, max_L):
    def collate(batch):
        B = len(batch)
        N = max(len(msa) for msa, _ in batch)
        tokens = torch.full((B, N, max_L), padding_idx, dtype=torch.long)
        labels = torch.zeros(B, N, dtype=torch.float)
        for b, (msa, lab) in enumerate(batch):
            for n, seq in enumerate(msa):
                ids = alphabet.encode(seq)[:max_L]
                tokens[b, n, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                labels[b, n] = float(lab[n])
        return tokens, labels
    return collate


def build_dataloader(csv_path, alphabet, padding_idx,
                     batch_size=4, n_msa=20, n_seq=25, seed=42):
    ds = BootstrapMSADataset(csv_path, n_msa=n_msa, n_seq=n_seq, seed=seed)
    max_L = max(len(s) for s in ds.sequences)
    print(f"[data] 全局最长序列 L = {max_L}")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=make_collate(alphabet, padding_idx, max_L),
    )
    return loader, max_L
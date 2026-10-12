import csv
import torch
import random
from torch.utils.data import Dataset, DataLoader

N_MSA = 20
N_SEQ_PER_MSA = 25

def load_csv(path):
    seqs, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seqs.append(row["sequence"])
            labels.append(int(row["label"]))
        idx = list(range(len(seqs)))
        random.shuffle(idx)
        seqs_shuffle = [seqs[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
    return seqs_shuffle, labels_shuffle


def group_msas(seqs, labels, n_msa=N_MSA):
    assert len(seqs) == len(labels) and len(seqs) % n_msa == 0
    n_per = len(seqs) // n_msa
    msas, msa_labels = [], []
    for i in range(n_msa):
        s, e = i * n_per, (i + 1) * n_per
        msas.append(seqs[s:e])
        msa_labels.append(labels[s:e])
    return msas, msa_labels


class MSADataset(Dataset):
    def __init__(self, csv_path, n_msa=20, n_seq=25):
        seqs, labels = load_csv(csv_path)
        self.msas, self.msa_labels = group_msas(seqs, labels, n_msa=n_msa)
        self.n_seq = n_seq

    def __len__(self):
        return len(self.msas)

    def __getitem__(self, idx):
        return self.msas[idx], self.msa_labels[idx]

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
                     batch_size=4, n_msa=20, n_seq=25, shuffle=True):
    ds = MSADataset(csv_path, n_msa=n_msa, n_seq=n_seq)
    max_L = max(len(s) for m in ds.msas for s in m)
    print(f"[data] 全局最长序列 L = {max_L}")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate(alphabet, padding_idx, max_L),
    )
    return loader, max_L
# generate_csv.py
import csv
import random

random.seed(42)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"

def generate_csv(path, n_sequences=500, min_len=50, max_len=150, gap_prob=0.1):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence", "label"])
        for i in range(n_sequences):
            L = random.randint(min_len, max_len)
            seq = "".join(
                GAP if random.random() < gap_prob else random.choice(AMINO_ACIDS)
                for _ in range(L)
            )
            writer.writerow([i, seq, random.randint(0, 1)])
    print(f"已生成 {n_sequences} 条序列 → {path}")


if __name__ == "__main__":
    generate_csv("data/random_msa.csv", n_sequences=500)
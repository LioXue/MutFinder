import csv

with open(r"C:\Users\Xuely\PycharmProjects\mutfinder\data\50_0.1_0_1.0_grande_msa.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = row["sequence"].strip()
        print(seq[69] if len(seq) >= 69 else "")
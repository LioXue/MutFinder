import csv
import random


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"

def generate_pseudo_dataset(
    csv_path: str,
    sequence: str,
    num: int,
    rate: float,
    loc_list: list[int],                #要标记的位置列表
    res_list: list[tuple[str, str]],    #每个位置对应的 (label0残基, label1残基)
    rate_list: list[tuple[float, float]],             #每个位置的标记比例 p
    seed: int = 42,
):

    L = len(sequence)

    assert 0.0 <= rate <= 1.0, f"rate={rate} 超出 [0, 1]"
    assert len(loc_list) == len(res_list) == len(rate_list), \
        f"loc_list/res_list/rate_list 长度必须一致：" \
        f"{len(loc_list)}/{len(res_list)}/{len(rate_list)}"
    for loc in loc_list:
        assert 0 <= loc < L, f"位置 {loc} 超出 [0, {L-1}]"
    for p0, p1 in rate_list:
        assert 0.0 <= min(p0,p1) <= max(p0,p1) <= 1.0, f"rate_list 中的 {p0,p1} 超出 [0, 1]"
    for r0, r1 in res_list:
        assert len(r0) == 1 and len(r1) == 1, \
            f"res_list 中每个 tuple 的两个元素都必须是单字符：{(r0, r1)}"

    random.seed(seed)
    n_mut = int(rate * L)
    num0 = num // 2
    labels = [0] * num0 + [1] * (num - num0)

    rows = []
    for label in labels:
        seq = list(sequence)

        positions = random.sample(range(L), n_mut)
        for p in positions:
            choices = [c for c in (AMINO_ACIDS + GAP) if c != seq[p]]
            seq[p] = random.choice(choices)

        rows.append([seq, label])

    for loc, (r0, r1), (p0, p1) in zip(loc_list, res_list, rate_list):
        n0 = int(p0 * num0)
        n1 = int(p1 * (num-num0))
        if n0 == 0 or n1 == 0:
            continue
        chose0 = random.sample(range(num0), n0)
        chose1 = random.sample(range(num0, num), n1)
        for i in chose0:
            seq, label = rows[i]
            seq[loc] = r0
        for i in chose1:
            seq, label = rows[i]
            seq[loc] = r1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        for seq, label in rows:
            writer.writerow(["".join(seq), label])

    return rows


if __name__ == "__main__":
    base_seq = (
        "MRRILSVLLENESGALSRVIGLFSQRGYNIESLTVAPTDDPTLSRMTIQTVGDEKVLEQIEKQLHKLVDVLRVSELGQGAHVEREIMLVKIQASGYGRDEVKRNTEIFRGQIIDVTPSLYTVQLAGTSGKLDAFLASIRDVAKIVEVARSGVVGLSRGDKIMR"
    )

    generate_pseudo_dataset(
        csv_path="data/50_0.1_0_1.0_grande_msa.csv",
        sequence=base_seq,
        num=50,
        rate=0.1,
        loc_list=[0],
        res_list=[("M", "A")],
        rate_list=[(1.0, 1.0)],
        seed=78,
    )
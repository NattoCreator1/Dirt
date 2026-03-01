from pathlib import Path
import numpy as np
import pandas as pd

def assign_level(score: float, b0: float, b1: float, b2: float) -> int:
    # 0=clean-ish, 1/2/3 increasing severity
    if score <= b0:
        return 0
    elif score <= b1:
        return 1
    elif score <= b2:
        return 2
    else:
        return 3

def main():
    proj = Path.home() / "soiling_project"
    meta = proj / "dataset/woodscape_processed/meta"
    splits = meta / "splits"

    labels_path = meta / "labels_index.csv"
    train_split_path = splits / "train_real.csv"

    df = pd.read_csv(labels_path)
    train_split = pd.read_csv(train_split_path)
    train_set = set(train_split["rel_path"].tolist())

    # 仅用 Train_Real 估计阈值，避免泄露
    df_tr = df[df["rel_img"].isin(train_set)].copy()

    # （可选）固定 clean 阈值：用 score 或用 soiling 面积
    # 推荐：用 global_soiling 做 b0 更稳定
    b0 = 0.01  # 对应 global_soiling<=1% 可视为近似干净
    df_tr_soiled = df_tr[df_tr["global_soiling"] > b0].copy()

    # 对非干净样本做分位数分箱（你可以选 0.33/0.66 或 0.30/0.70）
    q1, q2 = 0.33, 0.66
    b1 = float(df_tr_soiled["global_score"].quantile(q1))
    b2 = float(df_tr_soiled["global_score"].quantile(q2))

    print("Rebin thresholds:")
    print(f"  b0(clean by soiling) = {b0}")
    print(f"  b1(Q{q1})            = {b1}")
    print(f"  b2(Q{q2})            = {b2}")

    # 更新所有样本的 global_level/global_bin
    df["global_level"] = df.apply(
        lambda r: assign_level(float(r["global_score"]), b0=b0, b1=b1, b2=b2),
        axis=1
    )
    df["global_bin"] = (df["global_level"] > 0).astype(int)

    out_path = meta / "labels_index_rebinned.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

    # 打印新分布（便于确认不再塌缩）
    print("\nLevel distribution (all):")
    print(df["global_level"].value_counts().sort_index())
    print("\nLevel distribution (train_real):")
    print(df[df["rel_img"].isin(train_set)]["global_level"].value_counts().sort_index())

if __name__ == "__main__":
    main()

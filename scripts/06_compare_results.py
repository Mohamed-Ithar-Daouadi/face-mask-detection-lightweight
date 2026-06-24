# 06_compare_results.py
# Reads results/comparison.csv (produced by 05_evaluate_all.py) and generates
# the figures and summary table used in the report.
#
# Outputs (saved to results/figures/):
#   - accuracy_bar.png        mAP@0.5 and mAP@0.5:0.95 per model
#   - speed_bar.png           inference time (ms/image) per model
#   - size_bar.png            model size (MB) per model
#   - tradeoff_scatter.png    accuracy vs speed, point size = model size
#                             (this is the key figure: it visualizes the
#                              accuracy/speed trade-off the project studies)
#   - summary_table.md        a Markdown table to paste into the report

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = Path("results/comparison.csv")
OUT_DIR  = Path("results/figures")
# A consistent colour per model so every chart is easy to read together.
COLORS = {
    "YOLOv5n":       "#4C72B0",
    "YOLOv8n":       "#DD8452",
    "MobileNet-SSD": "#55A868",
}
# ──────────────────────────────────────────────────────────────────────────────


def load():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found — run 05_evaluate_all.py first."
        )
    df = pd.read_csv(CSV_PATH)
    # keep a stable model order for all charts
    order = ["YOLOv5n", "YOLOv8n", "MobileNet-SSD"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def colors_for(df):
    return [COLORS.get(m, "#888888") for m in df["model"]]


def bar_accuracy(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["map50"],   width=w, label="mAP@0.5",
           color=colors_for(df))
    ax.bar([i + w / 2 for i in x], df["map5095"], width=w, label="mAP@0.5:0.95",
           color=colors_for(df), alpha=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.set_title("Detection accuracy by model")
    # annotate values
    for i, (m50, m95) in enumerate(zip(df["map50"], df["map5095"])):
        ax.text(i - w / 2, m50 + 0.01, f"{m50:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, m95 + 0.01, f"{m95:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_bar.png", dpi=150)
    plt.close(fig)


def bar_speed(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["model"], df["ms_per_image"], color=colors_for(df))
    ax.set_ylabel("Inference time (ms / image)")
    ax.set_title("Inference speed by model (lower is better)")
    for i, v in enumerate(df["ms_per_image"]):
        ax.text(i, v + 0.1, f"{v:.2f} ms", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speed_bar.png", dpi=150)
    plt.close(fig)


def bar_size(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["model"], df["size_mb"], color=colors_for(df))
    ax.set_ylabel("Model size (MB)")
    ax.set_title("Model size by model (lower is better)")
    for i, v in enumerate(df["size_mb"]):
        ax.text(i, v + 0.05, f"{v:.1f} MB", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "size_bar.png", dpi=150)
    plt.close(fig)


def scatter_tradeoff(df):
    """
    The key figure: x = speed (ms), y = accuracy (mAP@0.5), bubble size = MB.
    The ideal model sits in the TOP-LEFT (fast + accurate). This makes the
    accuracy/speed trade-off visible at a glance.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        ax.scatter(
            row["ms_per_image"], row["map50"],
            s=row["size_mb"] * 40,                 # bubble area ~ model size
            color=COLORS.get(row["model"], "#888888"),
            alpha=0.75, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            f"{row['model']}\n({row['size_mb']:.1f} MB)",
            (row["ms_per_image"], row["map50"]),
            textcoords="offset points", xytext=(8, 8), fontsize=9,
        )
    ax.set_xlabel("Inference time (ms / image)  →  slower")
    ax.set_ylabel("mAP@0.5  →  more accurate")
    ax.set_title("Accuracy vs speed trade-off (bubble size = model size)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_scatter.png", dpi=150)
    plt.close(fig)


def write_table(df):
    lines = [
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms/img) | Size (MB) |",
        "|-------|---------|--------------|----------------|-----------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['map50']:.3f} | {r['map5095']:.3f} | "
            f"{r['ms_per_image']:.2f} | {r['size_mb']:.1f} |"
        )
    (OUT_DIR / "summary_table.md").write_text("\n".join(lines))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print("Loaded results:\n", df.to_string(index=False), "\n")

    bar_accuracy(df)
    bar_speed(df)
    bar_size(df)
    scatter_tradeoff(df)
    write_table(df)

    print(f"✅ Figures and table saved to {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
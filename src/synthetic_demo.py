"""Plot a schema-only demonstration using invented values (not model output)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/demo/synthetic_six_task_predictions.csv"
OUTPUT = ROOT / "outputs/synthetic_demo.png"
TASKS = [
    "FEV1_next", "FVC_next", "ratio_next",
    "interval_rapid_decline", "interval_FVC_rapid_decline",
    "interval_ratio_rapid_decline",
]
LABELS = ["Next FEV₁ (L)", "Next FVC (L)", "Next FEV₁/FVC",
          "Rapid FEV₁ decline", "Rapid FVC decline", "Rapid ratio decline"]
COLORS = {"Healthy": "#4C91B2", "COPD": "#DF8B72"}


def main():
    data = pd.read_csv(INPUT)
    required = {"population", "task", "observed", "predicted"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if set(data.task) != set(TASKS) or set(data.population) != set(COLORS):
        raise ValueError("The demo fixture must include each population/task exactly once.")
    if data.duplicated(["population", "task"]).any() or len(data) != 12:
        raise ValueError("Expected one synthetic row for each of 2 populations × 6 tasks.")

    fig, axes = plt.subplots(6, 2, figsize=(8, 12), constrained_layout=True)
    for j, (task, label) in enumerate(zip(TASKS, LABELS)):
        for i, pop in enumerate(COLORS):
            ax = axes[j, i]
            row = data[(data.population == pop) & (data.task == task)].iloc[0]
            ax.scatter(row.observed, row.predicted, color=COLORS[pop], s=50)
            if j < 3:
                lo = min(row.observed, row.predicted) - 0.1
                hi = max(row.observed, row.predicted) + 0.1
                ax.plot([lo, hi], [lo, hi], "--", color="#94A1AE", lw=1)
                ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Synthetic observed",
                       ylabel="Synthetic predicted")
            else:
                ax.set(xlim=(-0.15, 1.15), ylim=(-0.15, 1.15),
                       xticks=[0, 1], yticks=[0, 1],
                       xlabel="Synthetic label", ylabel="Illustrative score")
            ax.set_title(f"{'Healthy' if i == 0 else 'COPD'} | {label}", loc="left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Synthetic schema demo — invented values, not model predictions", fontsize=13)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(f"Validated {len(data)} synthetic rows; wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

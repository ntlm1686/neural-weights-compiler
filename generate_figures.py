"""Plot the saved, independently evaluated Stage 2 benchmark results.

Regenerate the data first with:
    python stage2_transformer_benchmark.py --output-json results/transformer_revalidation.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "results/transformer_revalidation.json").open(encoding="utf-8") as f:
        data = json.load(f)
    architectures = data["architectures"]
    tasks = list(next(iter(architectures.values())))
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, (name, results) in enumerate(architectures.items()):
        values = [100 * results[task]["accuracy"] for task in tasks]
        ax.bar(x + (i - 1) * 0.25, values, width=0.25, label=name)
    ax.set_xticks(x, [task.split('. ', 1)[-1] for task in tasks], rotation=35, ha="right")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Bounded-domain Transformer benchmarks ({data['trials_per_task']} cases per task)")
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "Shortest path uses 20 layers at each width; other tasks use the legend depth.",
             ha="center", fontsize=8)
    (ROOT / "assets").mkdir(exist_ok=True)
    fig.savefig(ROOT / "assets/transformer_revalidation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

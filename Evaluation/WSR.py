import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODEL = "jinppvit"
REFERENCE_VARIANT = "full"

VARIANTS = ["noencoder", "noskips", "nodecoder", "novit", "nomsa", "f4"]
METRICS = ["iou", "miss", "BIoU", "HD95", "MSD"]

BASE_DIR = os.path.join("Result", "prediction")

DATASET_NAME = os.path.basename(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
)

def load_vals(p):
    return np.loadtxt(p, delimiter=",").flatten()

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

for variant in VARIANTS:
    rows = []

    for metric in METRICS:
        ref = load_vals(os.path.join(BASE_DIR, MODEL, REFERENCE_VARIANT, f"{metric}.txt"))
        arr = load_vals(os.path.join(BASE_DIR, MODEL, variant, f"{metric}.txt"))

        m = min(len(ref), len(arr))
        W, p = wilcoxon(arr[:m], ref[:m])

        rows.append([
            metric,
            f"{W:.1f}",
            stars(p) if p < 0.05 else f"{p:.3e}"
        ])

    df = pd.DataFrame(rows, columns=["Metric", "W", "p-value"])

    tex = df.to_latex(
        escape=False,
        index=False,
        caption=f"Wilcoxon signed-rank tests comparing \\texttt{{jinppvit/{variant}}} vs \\texttt{{jinppvit/full}} on {DATASET_NAME}.",
        label=f"tab:wsr_{variant}_{DATASET_NAME}"
    )

    out_dir = os.path.join("Result", "WSR", variant)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{DATASET_NAME}.tex"), "w") as f:
        f.write(tex)

    print(f"[saved] WSR/{variant}/{DATASET_NAME}.tex")

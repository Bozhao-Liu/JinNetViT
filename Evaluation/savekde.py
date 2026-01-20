import os
import numpy as np

# -----------------------------------------
# CONFIG
# -----------------------------------------

MODEL = "jinppvit"

VARIANTS = [
    "full",
    "noencoder",
    "noskips",
    "nodecoder",
    "novit",
    "nomsa",
]

METRICS = ["iou", "miss", "BIoU", "HD95", "MSD"]

KDE_POINTS = 50

# -----------------------------------------
# KDE function
# -----------------------------------------

def compute_kde(arr, points=50):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.linspace(0, 1, points), np.zeros(points)

    # Detect bounds
    low = 0.0 if arr.min() >= 0 else arr.min()
    high = arr.max()

    xs = np.linspace(low, high, points)

    # Boundary correction for bounded metrics
    if low == 0.0 and high <= 1.0:
        arr_ref = np.concatenate([arr, -arr, 2 - arr])
    else:
        arr_ref = arr

    n = len(arr_ref)
    bw = np.std(arr_ref) * n ** (-1 / 5)
    bw = max(bw, 1e-6)

    diffs = (xs[:, None] - arr_ref[None, :]) / bw
    ys = np.exp(-0.5 * diffs**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))

    return xs, ys

# -----------------------------------------
# Main loop
# -----------------------------------------

for metric in METRICS:

    out_dir = f"Result/kde/{MODEL}/{metric}"
    os.makedirs(out_dir, exist_ok=True)

    for variant in VARIANTS:
        txt_path = f"Result/prediction/{MODEL}/{variant}/{metric}.txt"

        if not os.path.exists(txt_path):
            print(f"[skip] Missing: {txt_path}")
            continue

        vals = []
        with open(txt_path) as f:
            for tok in f.read().replace("\n", ",").split(","):
                try:
                    v = float(tok)
                    if v > 0:
                        vals.append(v)
                except:
                    pass

        xs, ys = compute_kde(vals, KDE_POINTS)

        dat_path = f"{out_dir}/{MODEL}{variant}.dat"
        with open(dat_path, "w") as f:
            for x, y in zip(xs, ys):
                f.write(f"{x:.6f} {y:.6f}\n")

        print(f"[saved] {dat_path}")

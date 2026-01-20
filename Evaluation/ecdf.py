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
    "f4"
]

METRICS = ["iou", "miss", "BIoU", "HD95", "MSD"]

ECDF_POINTS = 50

# -----------------------------------------
# ECDF function
# -----------------------------------------

def compute_ecdf(arr, points=50):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.linspace(0, 1, points), np.zeros(points)

    arr_sorted = np.sort(arr)
    xs = np.linspace(arr_sorted.min(), arr_sorted.max(), points)
    ys = np.searchsorted(arr_sorted, xs, side="right") / len(arr_sorted)

    return xs, ys

# -----------------------------------------
# Main loop
# -----------------------------------------

for metric in METRICS:

    out_dir = f"Result/ecdf/{MODEL}/{metric}"
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

        xs, ys = compute_ecdf(vals, ECDF_POINTS)

        dat_path = f"{out_dir}/{MODEL}{variant}.dat"
        with open(dat_path, "w") as f:
            for x, y in zip(xs, ys):
                f.write(f"{x:.6f} {y:.6f}\n")

        print(f"[saved] {dat_path}")

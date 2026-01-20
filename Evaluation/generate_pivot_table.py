import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# ============================================================
# CONFIGURATION (faithful to original intent)
# ============================================================

MODEL = "jinppvit"
REFERENCE_VARIANT = "full"

VARIANTS = [
    "noencoder",
    "noskips",
    "nodecoder",
    "novit",
    "nomsa",
    "f4"
]

METRICS = ["iou", "miss", "BIoU", "HD95", "MSD"]

base_pred_dir = os.path.join("Result", "prediction")

DATASET_NAME = os.path.basename(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
)

# ============================================================
# HELPERS (unchanged style)
# ============================================================

def load_vals(path):
    """Load a 1D array from .txt with flexible delimiters."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        try:
            arr = np.loadtxt(path)
        except ValueError:
            with open(path) as f:
                arr = [list(map(float, line.replace(",", " ").split())) for line in f]
            arr = np.array(arr)
    return arr.flatten()


def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def scale_matrix(mat, digits=1):
    """Same scaling logic as original WSR.py."""
    absmax = np.nanmax(np.abs(mat.values))
    if absmax == 0 or np.isnan(absmax):
        return mat.copy(), 0, ""
    exponent = int(np.floor(np.log10(absmax)))
    if exponent < 4:
        return mat.copy(), 0, ""
    scale = 10 ** exponent
    scaled = (mat / scale).round(digits)
    caption = f"(Values scaled by $10^{{{exponent}}}$.)"
    return scaled, exponent, caption

# ============================================================
# BUILD PIVOT TABLE (variants as rows)
# ============================================================

def build_pivot_table():

    # Column structure identical to original
    col_blocks = []
    for metric in METRICS:
        col_blocks += [
            f"{metric}_t", f"{metric}_tsig",
            f"{metric}_wsr", f"{metric}_wsrsig"
        ]

    df = pd.DataFrame(index=VARIANTS, columns=col_blocks)
    wsr_exponents = {}

    for metric in METRICS:

        # Load reference once
        ref_path = os.path.join(
            base_pred_dir, MODEL, REFERENCE_VARIANT, f"{metric}.txt"
        )
        if not os.path.exists(ref_path):
            raise RuntimeError(f"Missing reference file: {ref_path}")

        ref = load_vals(ref_path)

        # Per-variant statistics
        t_vals = {}
        t_sigs = {}
        w_vals = {}
        w_sigs = {}

        for variant in VARIANTS:
            var_path = os.path.join(
                base_pred_dir, MODEL, variant, f"{metric}.txt"
            )
            if not os.path.exists(var_path):
                t_vals[variant] = None
                t_sigs[variant] = ""
                w_vals[variant] = None
                w_sigs[variant] = ""
                continue

            arr = load_vals(var_path)
            m = min(len(arr), len(ref))
            a = arr[:m]
            b = ref[:m]

            # ---- paired t-test ----
            t, p = ttest_rel(a, b)
            t_vals[variant] = t
            t_sigs[variant] = f"$p={p:.3f}$" if p > 0.05 else stars(p)

            # ---- Wilcoxon ----
            try:
                W, p2 = wilcoxon(a, b, zero_method="wilcox", correction=False)
            except ValueError:
                W, p2 = np.nan, 1.0

            w_vals[variant] = W
            w_sigs[variant] = (
                f"$p={p2:.3f}$" if p2 > 0.05 else stars(p2)
            )

        # WSR scaling (metric-wise, same as original)
        W_df = pd.DataFrame(
            {v: w_vals[v] for v in VARIANTS},
            index=["W"]
        ).T

        W_scaled_df, exponent, _ = scale_matrix(W_df, digits=1)
        wsr_exponents[metric] = exponent

        # Fill table
        for variant in VARIANTS:
            tv = -t_vals[variant] if t_vals[variant] is not None else None
            df.loc[variant, f"{metric}_t"] = (
                "--" if tv is None else f"\\tcell{{{tv:.1f}}}"
            )
            df.loc[variant, f"{metric}_tsig"] = t_sigs[variant]

            wv = W_scaled_df.loc[variant, "W"]
            df.loc[variant, f"{metric}_wsr"] = (
                "--" if pd.isna(wv) else f"{wv:.1f}"
            )
            df.loc[variant, f"{metric}_wsrsig"] = w_sigs[variant]

    return df, wsr_exponents

# ============================================================
# LATEX EXPORT (same header logic as original)
# ============================================================

def to_latex(df, wsr_exponents):

    scaling_caption = []
    for metric in METRICS:
        exp = wsr_exponents[metric]
        if exp >= 4:
            scaling_caption.append(f"{metric}: $10^{{{exp}}}$")

    caption_tail = (
        "WSR scaled by " + ", ".join(scaling_caption) + "."
        if scaling_caption else
        "WSR values are unscaled."
    )

    header1 = "& "
    header2 = "& "
    for metric in METRICS:
        header1 += f"\\multicolumn{{4}}{{c|}}{{{metric}}} & "
        header2 += "t & Sig. & WSR & Sig. & "
    header1 = header1.rstrip("& ") + " \\\\"
    header2 = header2.rstrip("& ") + " \\\\"

    lines = []
    lines.append("% Auto-generated pivot table")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Component ablation results for JinPPViT, "
        "with paired $t$-test and Wilcoxon signed-rank comparisons "
        f"against the full model. {caption_tail}"
    )
    lines.append("\\begin{tabular}{l|" + "c" * (len(METRICS) * 4) + "}")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append("\\midrule")
    lines.append(header2)
    lines.append("\\midrule")

    for variant in df.index:
        row = variant
        for col in df.columns:
            row += " & " + str(df.loc[variant, col])
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

# ============================================================
# EXECUTION (single output file, same as original)
# ============================================================

df, wsr_exponents = build_pivot_table()
latex = to_latex(df, wsr_exponents)

out_dir = os.path.join("Result", "pivot_table")
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, f"{DATASET_NAME}.tex")
with open(out_file, "w") as f:
    f.write(latex)

print(f"Saved pivot table to:\n  {out_file}")

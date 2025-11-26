#!/usr/bin/env python
"""
plot_stl_run_on_atp.py
----------------------
Create one PDF per Case Name with four lines per metric:

    • STL‑mean  • STL‑min  • STL‑max  • ADP‑DEN
    • ADP‑DEN Equality  (if present)

Source CSV must contain at least:
  - Model Name   : 'STL', 'ADP‑DEN', …
  - RowType      : 'mean' | 'min' | 'max' | 'Equality' | NaN
  - Train Samples
  - one or more numeric metrics to plot
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------------------------------
# 1) Load & tidy
# ------------------------------------------------------------------
# Point to the file you uploaded into /mnt/data
CSV_PATH = "Results ran STL on ADP architectures\MIN_MAX_MEAN_STL_ADP.csv"
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# All numeric columns except the axes themselves are assumed to be metrics
numeric_cols = df.select_dtypes(include="number").columns.tolist()
metric_cols  = [c for c in numeric_cols
                if c not in ("Train Samples", "Test Samples", "Total_Neurons")]

# ------------------------------------------------------------------
# 2) Plot – one PDF per Case Name
# ------------------------------------------------------------------
OUT_DIR = "case_pdfs_adp-equality"
os.makedirs(OUT_DIR, exist_ok=True)

for case in df["Case Name"].unique():
    case_df = df[df["Case Name"] == case]

    safe_case = re.sub(r"[^A-Za-z0-9_\-]", "_", case)
    pdf_path  = os.path.join(OUT_DIR, f"{safe_case}_ADP-Equality.pdf")

    with PdfPages(pdf_path) as pdf:
        for metric in metric_cols:
            plt.figure(figsize=(8, 6))

            # -------- STL curves --------
            sty = dict(marker="o", linewidth=1, linestyle="--")


            rows = case_df.query("`Model Name` == 'STL' and RowType == 'mean'")        \
                            .sort_values("Train Samples")
            if not rows.empty:
                plt.plot(rows["Train Samples"], rows[metric],
                            label=f"STL‑{"mean"}", **sty)

            # -------- ADP‑DEN (main) ----
            adp_main = case_df.query("`Model Name` == 'ADP-DEN' and RowType.isna()")    \
                                .sort_values("Train Samples")
            if not adp_main.empty:
                plt.plot(adp_main["Train Samples"], adp_main[metric],
                         marker="s", linewidth=2.5, label="ADP‑DEN")

            # -------- ADP‑DEN Equality --
            adp_eq = case_df.query("`Model Name` == 'ADP-DEN' and RowType == 'Equality'")\
                            .sort_values("Train Samples")
            if not adp_eq.empty:
                plt.plot(adp_eq["Train Samples"], adp_eq[metric],
                         marker="D", linewidth=2.5,
                         label="ADP‑DEN Equality")

            # -------- cosmetics ----------
            plt.title(f"{metric} vs Train Samples\nCase: {case}", fontsize=11)
            plt.xlabel("Train Samples")
            plt.ylabel(metric)
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.legend()

            pdf.savefig()
            plt.close()

    print(f"✓  Wrote {pdf_path}")

print("\nDone – PDFs are in:", OUT_DIR)

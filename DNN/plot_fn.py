import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re, os

# ------------------------------------------------------------------
# 1) Load & tidy
# ------------------------------------------------------------------
CSV_PATH = "graphs_stlrunonadp.csv"        # ← adjust if needed
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------------
# 2) Identify what to plot
# ------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
metric_cols  = [c for c in numeric_cols if c not in ("Train Samples", "Test Samples")]

# Helper masks
is_stl = df["Model Name"].str.contains(r"STL", flags=re.IGNORECASE, regex=True)
is_adp = df["Model Name"].str.contains(r"ADP", flags=re.IGNORECASE, regex=True)

# ------------------------------------------------------------------
# 3) One PDF per Case Name
# ------------------------------------------------------------------
out_dir = "case_pdfs_stlrunonadp"
os.makedirs(out_dir, exist_ok=True)

for case in df["Case Name"].unique():
    case_df = df[df["Case Name"] == case]

    # Grab all distinct STL neuron counts present in *this* case
    # stl_neuron_values = (
    #     case_df[is_stl & (case_df["Case Name"] == case)] #["Total_Neurons"]
    #     # .dropna()
    #     # .unique()
    # )
    # stl_neuron_values = sorted(stl_neuron_values)

    safe_case = re.sub(r"[^A-Za-z0-9_\-]", "_", case)
    pdf_path  = os.path.join(out_dir, f"{safe_case}_adp_run_on_stl.pdf")

    with PdfPages(pdf_path) as pdf:
        for metric in metric_cols:
            plt.figure(figsize=(8, 6))

            # ---- ADP curve (single) ----
            adp_subset = (
                case_df[is_adp & (case_df["Case Name"] == case)]
                .sort_values("Train Samples")
            )
            if not adp_subset.empty:
                plt.plot(adp_subset["Train Samples"],
                         adp_subset[metric],
                         marker="o",
                         linewidth=2,
                         label="ADP")

            # ---- Multiple STL curves, one per Total_Neurons ----
            # for neurons in stl_neuron_values:
            stl_subset = (
                case_df[is_stl &
                        # (case_df["Total_Neurons"] == neurons) &
                        (case_df["Case Name"] == case)]
                .sort_values("Train Samples")
            )
            if not stl_subset.empty:
                plt.plot(stl_subset["Train Samples"],
                            stl_subset[metric],
                            linewidth=1,
                            marker="o",
                            label=f"STL run on ADP")
                    #  label=f"STL ({int(neurons)} neurons)")

            # ---- Formatting ----
            plt.title(f"{metric} vs Train Samples\nCase: {case}", fontsize=11)
            plt.xlabel("Number of Train Samples")
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            pdf.savefig()
            plt.close()

    print(f"✓  Wrote {pdf_path}")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

# Load the CSVs

mean_df = pd.read_csv("Results/mean_data.csv")
min_df = pd.read_csv("Results/min_data.csv")

# Convert "Time Taken" to seconds
def convert_to_seconds(t):
    if pd.isna(t):
        return None
    if isinstance(t, (int, float)):
        return float(t)
    m = re.match(r"(?:(\d+)m)?\s*(\d+\.\d+|\d+)s", t)
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2)) if m.group(2) else 0
        return minutes * 60 + seconds
    return None

mean_df["Time Taken (s)"] = mean_df["Time Taken"].apply(convert_to_seconds)
min_df["Time Taken (s)"] = min_df["Time Taken"].apply(convert_to_seconds)

# Identify columns to plot
meta_cols = ['Model Name', 'Case Name', 'Time Taken', 'Train Samples', 'Test Samples', 'Total_Neurons']
plot_cols = [col for col in mean_df.columns if col not in meta_cols and not col.endswith("(s)")]
plot_cols += ["Time Taken (s)", "Total_Neurons"]

# Function to create plot for a given dataframe and column
def plot_column(df, col, title_suffix):
    fig, ax = plt.subplots(figsize=(8, 5))
    stl_data = df[df["Model Name"].str.contains("STL")]
    for neurons in sorted(stl_data["Total_Neurons"].dropna().unique()[1:-1]):
        subset = stl_data[stl_data["Total_Neurons"] == neurons]
        ax.plot(subset["Train Samples"], subset[col], '--', label=f"STL-{int(neurons)}")

    den_data = df[df["Model Name"] == "ADP-DEN"]
    ax.plot(den_data["Train Samples"], den_data[col], '-', color='black', label="ADP-DEN")

    ax.set_title(f"{col} ({title_suffix})")
    ax.set_xlabel("Train Samples")
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True)
    return fig

# Output PDF
output_file = "MinMean_Plots.pdf"
with PdfPages(output_file) as pdf:
    for col in plot_cols:
        fig1 = plot_column(mean_df, col, "Mean")
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plot_column(min_df, col, "Min")
        pdf.savefig(fig2)
        plt.close(fig2)

print(f"PDF saved as: {output_file}")

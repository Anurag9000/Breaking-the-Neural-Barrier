import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_models_comparison(file_path: str, model_case_pairs: list, metrics: list, save_path_prefix: str):
    """
    Plots training samples vs. the specified metrics for one or more models on the same case.

    Args:
        file_path (str): Path to the CSV file.
        model_case_pairs (list): List of tuples (model_name, case_name).
        metrics (list): List of metric/column names to plot.
        save_path_prefix (str): Prefix to use for output PDF filenames.
    """
    data = pd.read_csv(file_path)

    for metric in metrics:
        if metric not in data.columns:
            print(f"\n❌ Metric '{metric}' not found in CSV columns. Available columns are:")
            print(data.columns.tolist())
            continue

        save_path = f"{save_path_prefix}_{metric.replace(' ', '_')}.pdf"

        with PdfPages(save_path) as pdf:
            # Plot 1: Linear scale
            plt.figure(figsize=(10, 6))
            for model_name, case_name in model_case_pairs:
                df = data[(data['Model Name'] == model_name) & (data['Case Name'] == case_name)]
                if df.empty:
                    print(f"[Warning] No data for model: {model_name}, case: {case_name}")
                    continue
                plt.plot(df['Train Samples'], df[metric], marker='o', label=f"{model_name} ({case_name})")
            plt.title(f"Training Samples vs. {metric} (Linear Scale)")
            plt.xlabel("Training Samples")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

            # Plot 2: Log scale (Y axis)
            plt.figure(figsize=(10, 6))
            for model_name, case_name in model_case_pairs:
                df = data[(data['Model Name'] == model_name) & (data['Case Name'] == case_name)]
                if df.empty:
                    continue
                plt.plot(df['Train Samples'], df[metric], marker='o', label=f"{model_name} ({case_name})")
            plt.yscale('log')
            plt.title(f"Training Samples vs. {metric} (Log Scale)")
            plt.xlabel("Training Samples")
            plt.ylabel(f"{metric} [Log Scale]")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            pdf.savefig()
            plt.close()

        print(f"✅ Plots for metric '{metric}' saved at: {save_path}")


# ---- Interactive Prompt ----
if __name__ == "__main__":
    print("How many models do you want to compare? (1 or more):")
    try:
        num_models = int(input().strip())
    except ValueError:
        print("Invalid number. Exiting.")
        exit(1)

    model_case_pairs = []
    for i in range(num_models):
        print(f"\nEnter model name #{i+1} (e.g., STL, DEN, MTL):")
        model_name = input().strip()
        print(f"Enter corresponding case name for model #{i+1} (e.g., pglib_opf_case14_ieee):")
        case_name = input().strip()
        model_case_pairs.append((model_name, case_name))

    print("\nEnter the column name to plot (e.g., Mean MSE, Constraint Violation, etc.) or type 'all' to plot all numeric metrics:")
    metric_input = input().strip()

    summary_df = pd.read_csv("Results/SUMMARY.csv")
    numeric_cols = summary_df.select_dtypes(include='number').columns.tolist()
    available_cols = [col for col in numeric_cols if col != 'Train Samples']

    if metric_input.lower() == 'all':
        metrics_to_plot = available_cols
    else:
        metrics_to_plot = [metric_input]

    model_names = "_vs_".join([m for m, _ in model_case_pairs])
    case_names = "_".join(set([c for _, c in model_case_pairs]))
    save_prefix = f"Results/{model_names}_{case_names}"

    plot_models_comparison(
        file_path="Results/SUMMARY.csv",
        model_case_pairs=model_case_pairs,
        metrics=metrics_to_plot,
        save_path_prefix=save_prefix
    )

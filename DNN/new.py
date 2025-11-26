
from pathlib import Path
import pandas as pd

def process_summary_data(summary_csv_path: str):
        if not Path(summary_csv_path).exists():
            print(f"SUMMARY.csv not found at {summary_csv_path}")
            return

        # Load the SUMMARY.csv file
        summary_data = pd.read_csv(summary_csv_path)

        # Define paths for min_data and mean_data
        min_data_path = summary_csv_path.replace("SUMMARY.csv", "min_data.csv")
        mean_data_path = summary_csv_path.replace("SUMMARY.csv", "mean_data.csv")

        # Split ADP-DEN and other models
        adp_data = summary_data[summary_data['Model Name'] == 'ADP-DEN']
        other_data = summary_data[summary_data['Model Name'] != 'ADP-DEN']

        # Min Data
        min_adp = adp_data.loc[
            adp_data.groupby(['Model Name', 'Case Name', 'Train Samples', 'Test Samples'])['Mean MSE'].idxmin()
        ]
        min_other = other_data.loc[
            other_data.groupby(['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'])['Mean MSE'].idxmin()
        ]
        min_data = pd.concat([min_adp, min_other])

        if Path(min_data_path).exists():
            existing_min_data = pd.read_csv(min_data_path)
            min_data = pd.concat([existing_min_data, min_data]).drop_duplicates(
                subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'],
                keep='last'
            )
        min_data.to_csv(min_data_path, index=False)
        print(f"Min Data saved to {min_data_path}")

        # Mean Data
        mean_adp = adp_data.groupby(
            ['Model Name', 'Case Name', 'Train Samples', 'Test Samples'], as_index=False
        ).mean(numeric_only=True)
        mean_other = other_data.groupby(
            ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'], as_index=False
        ).mean(numeric_only=True)

        # Retain non-numeric data to merge back
        adp_keys = ['Model Name', 'Case Name', 'Train Samples', 'Test Samples']
        other_keys = ['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons']

        adp_non_num = adp_data[adp_keys].drop_duplicates()
        other_non_num = other_data[other_keys].drop_duplicates()

        mean_adp = pd.merge(adp_non_num, mean_adp, on=adp_keys, how='inner')
        mean_other = pd.merge(other_non_num, mean_other, on=other_keys, how='inner')

        mean_data = pd.concat([mean_adp, mean_other])

        if Path(mean_data_path).exists():
            existing_mean_data = pd.read_csv(mean_data_path)
            mean_data = pd.concat([existing_mean_data, mean_data]).drop_duplicates(
                subset=['Model Name', 'Case Name', 'Train Samples', 'Test Samples', 'Total_Neurons'],
                keep='last'
            )
        mean_data.to_csv(mean_data_path, index=False)
        print(f"Mean Data saved to {mean_data_path}")

# Run the function
summary_csv_path = "Results Case500/SUMMARY.csv"
process_summary_data(summary_csv_path)

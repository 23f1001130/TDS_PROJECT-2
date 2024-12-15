
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "tabulate",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def summarize_dataset(data):
    print("Generating dataset summary...")
    stats = data.describe()
    missing = data.isnull().sum()
    numeric_data = data.select_dtypes(include=[np.number])
    correlations = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()
    print("Summary generated.")
    return stats, missing, correlations

def identify_outliers(data):
    print("Detecting anomalies...")
    numeric_cols = data.select_dtypes(include=[np.number])
    q1 = numeric_cols.quantile(0.25)
    q3 = numeric_cols.quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((numeric_cols < (q1 - 1.5 * iqr)) | (numeric_cols > (q3 + 1.5 * iqr))).sum()
    print("Anomaly detection complete.")
    return outlier_counts

def create_visualizations(correlation_matrix, anomaly_counts, data, save_directory):
    print("Creating data visualizations...")
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    heatmap_path = os.path.join(save_directory, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    if not anomaly_counts.empty and anomaly_counts.sum() > 0:
        plt.figure(figsize=(6, 6))
        anomaly_counts.plot(kind='bar', color='orange')
        plt.title('Outlier Analysis')
        plt.xlabel('Columns')
        plt.ylabel('Outlier Count')
        outlier_plot_path = os.path.join(save_directory, 'outlier_analysis.png')
        plt.savefig(outlier_plot_path)
        plt.close()
    else:
        outlier_plot_path = None

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if numeric_cols.any():
        first_col = numeric_cols[0]
        plt.figure(figsize=(6, 6))
        sns.histplot(data[first_col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {first_col}')
        dist_plot_path = os.path.join(save_directory, f'{first_col}_distribution.png')
        plt.savefig(dist_plot_path)
        plt.close()
    else:
        dist_plot_path = None

    print("Visualizations created successfully.")
    return heatmap_path, outlier_plot_path, dist_plot_path

def generate_report(stats, missing_vals, correlations, anomalies, save_directory, dist_plot_path=None):
    print("Compiling report...")
    report_path = os.path.join(save_directory, 'ANALYSIS_REPORT.md')
    with open(report_path, 'w') as file:
        file.write("# Data Analysis Summary\n\n")
        file.write("## Overview\nThis report presents insights from the analyzed dataset, along with visualizations of key patterns and anomalies.\n\n")
        file.write("## Summary Statistics\n")
        file.write(stats.to_markdown(tablefmt="grid"))
        file.write("\n\n")
        file.write("## Missing Data\n")
        file.write(missing_vals.to_markdown(tablefmt="grid"))
        file.write("\n\n")
        file.write("## Outliers\n")
        file.write(anomalies.to_markdown(tablefmt="grid"))
        file.write("\n\n")
        file.write("## Correlation Matrix\n")
        file.write("![](correlation_heatmap.png)\n\n")
        if anomalies.sum() > 0:
            file.write("## Outlier Analysis\n")
            file.write("![](outlier_analysis.png)\n\n")
        if dist_plot_path:
            file.write("## Distribution Analysis\n")
            file.write(f"![Distribution](./{os.path.basename(dist_plot_path)})\n\n")
        file.write("---\n## Notes\nThis report was automatically generated and reflects data insights as analyzed.\n")
    print("Report generated: ", report_path)
    return report_path

def run_analysis(data_path):
    print("Initializing data analysis pipeline...")
    try:
        print(f"Attempting to load file: {data_path}")
        # Try loading the file
        data = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
        print("Data loaded successfully.")
        print("First few rows of the dataset:\n", data.head())
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        print("Trying alternate delimiters...")
        try:
            data = pd.read_csv(data_path, encoding='utf-8', delimiter=';', encoding_errors='replace')
            print("Data loaded successfully with alternate delimiter.")
        except Exception as e:
            print(f"Error loading data with alternate delimiter: {e}")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    stats, missing, correlations = summarize_dataset(data)
    anomalies = identify_outliers(data)

    output_directory = "results"
    heatmap_path, outlier_plot_path, dist_plot_path = create_visualizations(correlations, anomalies, data, output_directory)
    generate_report(stats, missing, correlations, anomalies, output_directory, dist_plot_path)
    print("Pipeline execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis Tool")
    parser.add_argument("file", nargs='?', default="media.csv", help="Path to the dataset (CSV file)")
    args = parser.parse_args()
    run_analysis(args.file)

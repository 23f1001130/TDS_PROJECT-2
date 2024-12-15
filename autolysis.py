# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "matplotlib",
#   "pandas",
#   "requests",
#   "openai",
#   "ipykernel",
#   "tabulate",
#   "importlib",
#    "pandas",
#   "chardet"  # Add all packages used in the script
# ]


import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---
def compute_summary_statistics(df):
    """Generate descriptive statistics for the dataset."""
    return df.describe(include='all').fillna('N/A')

def detect_outliers(df, col):
    """Detect outliers using the Interquartile Range (IQR) method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    return outliers if not outliers.empty else pd.DataFrame({'Message': ['No outliers detected']})

def plot_correlation_matrix(df, output_path):
    """Generate and save a correlation matrix heatmap."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(output_path)
    plt.close()

def generate_narrative(summary_stats, outliers, dataset_name):
    """Generate a narrative report for the dataset."""
    narrative = f"# Report for {dataset_name}\n\n"
    narrative += "## Summary Statistics\n" + summary_stats.to_markdown() + "\n\n"
    if not outliers.empty:
        narrative += "## Outliers Detected\n" + outliers.to_markdown() + "\n\n"
    else:
        narrative += "## Outliers Detected\nNo significant outliers found.\n\n"
    return narrative

def generate_visualizations(df, dataset_name, output_dir):
    """Generate various visualizations for the dataset."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Pair plot
    pairplot_path = os.path.join(output_dir, f"{dataset_name}_pairplot.png")
    sns.pairplot(df[numeric_columns])
    plt.savefig(pairplot_path)
    plt.close()

    # Boxplot for outliers
    boxplot_path = os.path.join(output_dir, f"{dataset_name}_boxplot.png")
    df[numeric_columns].plot(kind='box', figsize=(6,6))
    plt.title("Boxplot of Numeric Features")
    plt.savefig(boxplot_path)
    plt.close()

# --- Main Processing Function ---
def process_dataset(file_path, output_dir):
    """Process a dataset: analyze, visualize, and save outputs."""
    # Load the dataset
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    narrative_path = os.path.join(output_dir, f"{dataset_name}_report.md")
    visualization_path = os.path.join(output_dir, f"{dataset_name}_correlation.png")

    # Perform analysis
    summary_stats = compute_summary_statistics(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = pd.concat([detect_outliers(df, col) for col in numeric_columns])

    # Generate visualizations
    if not df[numeric_columns].empty:
        plot_correlation_matrix(df[numeric_columns], visualization_path)
        generate_visualizations(df, dataset_name, output_dir)

    # Generate narrative
    narrative = generate_narrative(summary_stats, outliers, dataset_name)

    # Save narrative
    with open(narrative_path, "w") as f:
        f.write(narrative)

    print(f"Analysis complete for {dataset_name}. Outputs saved in {output_dir}")

# --- Main Function ---
def main(csv_file, output_dir="results"):
    """Main entry point for analysis."""
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        print(f"Processing dataset: {csv_file}")
        print(f"Saving results to: {output_dir}")
        process_dataset(csv_file, output_dir)
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def process_dataset(csv_file, output_dir):
    # Add your dataset processing logic here
    print(f"Dataset {csv_file} is being processed. Results will be saved in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autolysis script for CSV analysis.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results",  # Default value for output_dir
        help="Path to the output directory (default: 'results')."
    )
    args = parser.parse_args()
    main(args.csv_file, args.output_dir)

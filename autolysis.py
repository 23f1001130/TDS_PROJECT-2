
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "matplotlib",
#   "pandas",
#   "requests",
#   "openai",
#   "ipykernel",
#    "tabulate",
#    "importlib",
#   "chardet"  # Add all packages used in the script
# ]
# ///
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json


def data_summary(dataframe):
    """
    Generate a summary of the data, including descriptive statistics, missing values, 
    and correlation matrix for numeric columns.

    Args:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        tuple: Summary statistics, missing values, and correlation matrix.
    """
    summary_stats = dataframe.describe()
    missing_values = dataframe.isnull().sum()
    numeric_data = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()
    
    return summary_stats, missing_values, correlation_matrix


def identify_outliers(dataframe):
    """
    Identify potential outliers in numeric columns using the Interquartile Range (IQR) method.

    Args:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        pd.Series: A count of outliers for each numeric column.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return pd.Series(dtype=int)  # Return an empty Series if no numeric columns exist

    q1, q3 = numeric_data.quantile(0.25), numeric_data.quantile(0.75)
    iqr = q3 - q1
    outlier_flags = (numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))
    
    return outlier_flags.sum()


def create_visualizations(dataframe, correlation_matrix, output_dir):
    """
    Generate visualizations: Correlation heatmap, outlier count bar chart, and distribution plot.

    Args:
        dataframe (pd.DataFrame): The input dataset.
        correlation_matrix (pd.DataFrame): The correlation matrix.
        output_dir (str): Path to save the visualizations.

    Returns:
        list: Paths to the saved visualization files.
    """
    visualizations = []
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Matrix Heatmap
    if not correlation_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        heatmap_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.title('Correlation Matrix')
        plt.savefig(heatmap_path)
        plt.close()
        visualizations.append(heatmap_path)

    # Distribution Plot for Numeric Data
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[numeric_columns[0]], kde=True, color='blue', bins=30)
        dist_plot_path = os.path.join(output_dir, 'distribution_plot.png')
        plt.title('Data Distribution')
        plt.savefig(dist_plot_path)
        plt.close()
        visualizations.append(dist_plot_path)

    return visualizations


def save_analysis_report(output_dir, summary_stats, missing_values, outliers):
    """
    Save an analysis report with summary statistics, missing values, and outliers to a README.md file.

    Args:
        output_dir (str): Directory to save the README file.
        summary_stats (pd.DataFrame): Descriptive statistics.
        missing_values (pd.Series): Missing values count.
        outliers (pd.Series): Outliers count.

    Returns:
        str: Path to the README file.
    """
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")
        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")
        f.write("## Outlier Detection\n")
        f.write(outliers.to_markdown() + "\n\n")
    return readme_path


def main(csv_file, output_dir="."):
    """
    Main function to perform data analysis, visualization, and reporting.

    Args:
        csv_file (str): Path to the input CSV file.
        output_dir (str): Path to save the analysis outputs.
    """
    # Load the dataset
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Perform analysis
    summary_stats, missing_values, corr_matrix = data_summary(df)
    outliers = identify_outliers(df)

    # Generate visualizations
    visualizations = create_visualizations(df, corr_matrix, output_dir)

    # Save analysis report
    readme_path = save_analysis_report(output_dir, summary_stats, missing_values, outliers)

    print(f"Analysis complete. Report: {readme_path}")
    print(f"Visualizations: {visualizations}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated data analysis tool.")
    parser.add_argument("csv_file", help="Path to the CSV file for analysis.")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs.")
    args = parser.parse_args()

    main(args.csv_file, args.output_dir)

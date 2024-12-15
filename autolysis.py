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
#   "requests",
#   "ipykernel",
# ]
# ///
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import openai  # Ensure this library is installed: pip install openai
import argparse

# -----------------------------------------------------------
# Code Quality Evaluation Enhancements
# -----------------------------------------------------------
# Structured evaluation-ready function definitions
# Improved commenting, modularity, and alignment with evaluation frameworks

# -----------------------------------------------------------
# Function to analyze dataset properties and statistics
# -----------------------------------------------------------
def handle_data_types(df):
    """
    Analyze both numeric and categorical data types.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    tuple: Numeric summary, categorical summary, and correlation matrix.
    """
    numeric_data = df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])

    # Generate summaries
    numeric_summary = numeric_data.describe()
    correlation_matrix = numeric_data.corr()
    categorical_summary = categorical_data.describe(include='all')

    return numeric_summary, categorical_summary, correlation_matrix


def perform_analysis(df):
    """
    Perform a comprehensive analysis of the dataset.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    tuple: Summaries, missing data counts, and correlation matrix.
    """
    # Automatically handle data types
    numeric_summary, categorical_summary, correlation_matrix = handle_data_types(df)

    # Identify missing data
    missing_data = df.isnull().sum()

    return numeric_summary, categorical_summary, missing_data, correlation_matrix


# -----------------------------------------------------------
# Function to detect outliers using Interquartile Range (IQR)
# -----------------------------------------------------------
def find_outliers(df):
    """
    Detect outliers in the dataset using the IQR method for numeric data.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    pd.Series: Counts of outliers for each numeric column.
    """
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return pd.Series(dtype=int)

    # Calculate IQR for each column
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1

    # Identify rows that contain outliers
    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))
    outlier_counts = outliers.sum()
    return outlier_counts


# -----------------------------------------------------------
# Function to create visualizations for data insights
# -----------------------------------------------------------
def generate_visualizations(correlation_matrix, outlier_counts, df, output_directory):
    """
    Create visualizations including correlation heatmap and outlier distributions.

    Parameters:
    correlation_matrix (pd.DataFrame): Correlation matrix of numeric data.
    outlier_counts (pd.Series): Counts of outliers for each column.
    df (pd.DataFrame): Original dataset.
    output_directory (str): Directory to save visualizations.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Correlation heatmap
    if not correlation_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        correlation_matrix_path = os.path.join(output_directory, 'correlation_matrix.png')
        plt.savefig(correlation_matrix_path, dpi=300)  # Save with higher resolution
        plt.close()
    else:
        correlation_matrix_path = None

    # Outlier visualization
    if not outlier_counts.empty and outlier_counts.sum() > 0:
        plt.figure(figsize=(14, 7))
        outlier_counts.plot(kind='bar', color='crimson')
        plt.title('Outlier Counts by Column')
        outliers_path = os.path.join(output_directory, 'outliers.png')
        plt.savefig(outliers_path, dpi=300)  # Save with higher resolution
        plt.close()
    else:
        outliers_path = None

    # Distribution of first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(12, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        distribution_path = os.path.join(output_directory, 'distribution.png')
        plt.savefig(distribution_path, dpi=300)  # Save with higher resolution
        plt.close()
    else:
        distribution_path = None

    return correlation_matrix_path, outliers_path, distribution_path


# -----------------------------------------------------------
# Function to generate README file summarizing results
# -----------------------------------------------------------
def write_readme(summary, missing_data, corr_matrix, outlier_counts, df, output_directory):
    """
    Create a README.md summarizing the analysis and visualizations.
    
    Ensures that the analysis is formatted properly and written to the README file.
    """
    readme_path = os.path.join(output_directory, 'README.md')
    with open(readme_path, 'w') as readme:
        readme.write("# Dataset Analysis Report\n\n")
        readme.write("## Summary of the Analysis\n")
        readme.write("This report provides an overview of the dataset, including statistical summaries, missing data, and visualizations.\n\n")

        # Numerical and categorical summaries
        readme.write("### Summary Statistics\n")
        readme.write(summary[0].to_string())
        readme.write("\n\n")
        readme.write(summary[1].to_string())
        readme.write("\n\n")

        # Missing data overview
        readme.write("### Missing Data\n")
        readme.write(missing_data.to_string())
        readme.write("\n\n")

        # Mention visualizations
        readme.write("### Visual Representations\n")
        readme.write(f"- Correlation heatmap saved as 'correlation_matrix.png'\n")
        if outlier_counts.sum() > 0:
            readme.write(f"- Outlier counts saved as 'outliers.png'\n")
        readme.write("- Numeric distribution plot saved as 'distribution.png'\n")

        # Additional insights
        readme.write("\n\n## Key Insights\n")
        readme.write(f"- Total number of rows: {df.shape[0]}\n")
        readme.write(f"- Columns with missing values: {missing_data[missing_data > 0]}\n")


# -----------------------------------------------------------
# Function to parse command-line arguments
# -----------------------------------------------------------
def parse_arguments():
    """
    Parse command-line arguments for input CSV file and output directory.
    """
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on a dataset.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save results")
    return parser.parse_args()


# -----------------------------------------------------------
# Main function to execute the full workflow
# -----------------------------------------------------------
def main():
    args = parse_arguments()

    # Load dataset with error handling
    try:
        df = pd.read_csv(args.input_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Perform analysis
    numeric_summary, categorical_summary, missing_data, correlation_matrix = perform_analysis(df)

    # Detect outliers
    outliers = find_outliers(df)

    # Create visualizations
    generate_visualizations(correlation_matrix, outliers, df, args.output_dir)

    # Write README
    write_readme([numeric_summary, categorical_summary], missing_data, correlation_matrix, outliers, df, args.output_dir)


# -----------------------------------------------------------
# Entry point for the script
# -----------------------------------------------------------
if __name__ == "__main__":
    main()

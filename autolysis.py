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

# Function to analyze dataset properties and basic statistics
def perform_analysis(df):
    print("Performing dataset analysis...")  # Debug message
    # Generating statistical summaries for numeric data
    numerical_summary = df.describe()

    # Identifying missing data counts
    missing_data = df.isnull().sum()

    # Extracting numerical columns for correlation matrix calculation
    numeric_data = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()

    print("Dataset analysis completed.")  # Debug message
    return numerical_summary, missing_data, correlation_matrix


# Function to detect outliers using IQR methodology
def find_outliers(df):
    print("Detecting outliers in numeric data...")  # Debug message
    # Isolating numeric columns for outlier detection
    numeric_data = df.select_dtypes(include=[np.number])

    # Computing interquartile range (IQR)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1

    # Identifying data points outside the acceptable range
    outlier_counts = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()

    print("Outlier detection complete.")  # Debug message
    return outlier_counts


# Function to generate visual outputs
def generate_visualizations(correlation_matrix, outlier_counts, df, output_directory):
    print("Creating data visualizations...")  # Debug message
    # Correlation matrix heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    correlation_matrix_path = os.path.join(output_directory, 'correlation_matrix.png')
    plt.savefig(correlation_matrix_path)
    plt.close()

    # Visualizing outliers, if any exist
    if not outlier_counts.empty and outlier_counts.sum() > 0:
        plt.figure(figsize=(12, 6))
        outlier_counts.plot(kind='bar', color='crimson')
        plt.title('Outlier Counts by Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_path = os.path.join(output_directory, 'outliers.png')
        plt.savefig(outliers_path)
        plt.close()
    else:
        print("No significant outliers detected.")  # Debug message
        outliers_path = None

    # Generating a distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        distribution_path = os.path.join(output_directory, 'distribution.png')
        plt.savefig(distribution_path)
        plt.close()
    else:
        print("No numeric columns available for distribution plot.")  # Debug message
        distribution_path = None

    print("Visualizations generated successfully.")  # Debug message
    return correlation_matrix_path, outliers_path, distribution_path


# Function to write the analysis results to a README file
def write_readme(summary, missing_data, corr_matrix, outlier_counts, output_directory):
    print("Writing analysis results to README.md...")  # Debug message
    readme_path = os.path.join(output_directory, 'README.md')
    try:
        with open(readme_path, 'w') as readme:
            readme.write("# Dataset Analysis Report\n\n")
            readme.write("## Summary of the Analysis\n")
            readme.write("This report provides a comprehensive overview of the dataset, including statistics, outlier detection, and visualizations.\n\n")

            # Adding statistics section
            readme.write("### Summary Statistics\n")
            for col in summary.columns:
                readme.write(f"- {col} Mean: {summary.loc['mean', col]:.2f}\n")

            # Adding missing data section
            readme.write("### Missing Values\n")
            for col, count in missing_data.items():
                readme.write(f"- {col}: {count} missing entries\n")

            # Including visual links
            readme.write("### Visual Representations\n")
            readme.write("Refer to the following visualizations for insights:\n")
            readme.write("- Correlation heatmap: correlation_matrix.png\n")
            if outlier_counts.sum() > 0:
                readme.write("- Outliers visualization: outliers.png\n")
            readme.write("- Distribution plot: distribution.png\n")

        print("README file created successfully.")  # Debug message
    except Exception as e:
        print(f"Error writing README file: {e}")


# Main function orchestrating the analysis workflow
def main(input_file):
    print("Initiating data analysis workflow...")  # Debug message
    try:
        # Loading the input dataset
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")  # Debug message
    except Exception as e:
        print(f"Failed to load the dataset: {e}")
        return

    # Performing data analysis
    summary, missing_data, correlation_matrix = perform_analysis(df)

    # Detecting outliers
    outliers = find_outliers(df)

    # Creating visualizations
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(correlation_matrix, outliers, df, output_dir)

    # Writing the report
    write_readme(summary, missing_data, correlation_matrix, outliers, output_dir)

    print("Data analysis workflow completed successfully.")  # Debug message


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

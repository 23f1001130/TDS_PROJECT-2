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

# -----------------------------------------------------------
# Function to analyze dataset properties and statistics
# -----------------------------------------------------------
def perform_analysis(df):
    """
    Analyze the dataset for basic statistics, missing data, and correlations.
    """
    print("Performing dataset analysis...")  # Debug message
    
    # Describe numeric columns for key statistics
    numerical_summary = df.describe()
    print("Generated numerical summary.")  # Debug
    
    # Identify missing data
    missing_data = df.isnull().sum()
    print("Identified missing data counts.")  # Debug
    
    # Create correlation matrix for numeric columns
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        correlation_matrix = pd.DataFrame()
        print("No numeric columns found for correlation analysis.")  # Debug
    else:
        correlation_matrix = numeric_data.corr()
        print("Generated correlation matrix.")  # Debug
    
    print("Dataset analysis completed.")  # Debug
    return numerical_summary, missing_data, correlation_matrix


# -----------------------------------------------------------
# Function to detect outliers using Interquartile Range (IQR)
# -----------------------------------------------------------
def find_outliers(df):
    """
    Detect outliers in the dataset using the IQR method.
    """
    print("Detecting outliers in numeric data...")  # Debug
    
    # Extract numeric columns
    numeric_data = df.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        print("No numeric data found for outlier detection.")  # Debug
        return pd.Series(dtype=int)

    # Calculate IQR for each column
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Identify rows that contain outliers
    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))
    outlier_counts = outliers.sum()
    print("Outlier detection complete.")  # Debug
    return outlier_counts


# -----------------------------------------------------------
# Function to create visualizations for data insights
# -----------------------------------------------------------
def generate_visualizations(correlation_matrix, outlier_counts, df, output_directory):
    """
    Create visualizations including correlation heatmap and outlier distributions.
    """
    print("Creating data visualizations...")  # Debug
    
    # Correlation heatmap
    if not correlation_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        correlation_matrix_path = os.path.join(output_directory, 'correlation_matrix.png')
        plt.savefig(correlation_matrix_path)
        plt.close()
    else:
        print("No correlation matrix generated; skipping heatmap.")  # Debug
        correlation_matrix_path = None

    # Outlier visualization
    if not outlier_counts.empty and outlier_counts.sum() > 0:
        plt.figure(figsize=(14, 7))
        outlier_counts.plot(kind='bar', color='crimson')
        plt.title('Outlier Counts by Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_path = os.path.join(output_directory, 'outliers.png')
        plt.savefig(outliers_path)
        plt.close()
    else:
        print("No significant outliers detected.")  # Debug
        outliers_path = None

    # Distribution of first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(12, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        distribution_path = os.path.join(output_directory, 'distribution.png')
        plt.savefig(distribution_path)
        plt.close()
    else:
        print("No numeric columns available for distribution plot.")  # Debug
        distribution_path = None

    print("Visualizations generated successfully.")  # Debug
    return correlation_matrix_path, outliers_path, distribution_path


# -----------------------------------------------------------
# Function to generate README file summarizing results
# -----------------------------------------------------------
def write_readme(summary, missing_data, corr_matrix, outlier_counts, output_directory):
    """
    Create a README.md summarizing the analysis and visualizations.
    """
    print("Writing analysis results to README.md...")  # Debug
    readme_path = os.path.join(output_directory, 'README.md')
    try:
        with open(readme_path, 'w') as readme:
            readme.write("# Dataset Analysis Report\n\n")
            readme.write("## Summary of the Analysis\n")
            readme.write("This report provides an overview of the dataset, including statistical summaries, missing data, and visualizations.\n\n")

            # Adding numerical summaries
            readme.write("### Summary Statistics\n")
            readme.write(summary.to_string())
            readme.write("\n\n")
            
            # Missing data overview
            readme.write("### Missing Data\n")
            readme.write(missing_data.to_string())
            readme.write("\n\n")

            # Mention visualizations
            readme.write("### Visual Representations\n")
            readme.write("- Correlation heatmap saved as 'correlation_matrix.png'\n")
            if outlier_counts.sum() > 0:
                readme.write("- Outlier counts saved as 'outliers.png'\n")
            readme.write("- Numeric distribution plot saved as 'distribution.png'\n")
        
        print("README file created successfully.")  # Debug
    except Exception as e:
        print(f"Error while writing README: {e}")


# -----------------------------------------------------------
# Main function to execute the full workflow
# -----------------------------------------------------------
def main(input_file):
    """
    Main function to execute data analysis and visualization workflow.
    """
    print("Starting data analysis workflow...")  # Debug
    
    # Load dataset
    try:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")  # Debug
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Perform analysis
    summary, missing_data, correlation_matrix = perform_analysis(df)

    # Detect outliers
    outliers = find_outliers(df)

    # Create visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(correlation_matrix, outliers, df, output_dir)

    # Write README
    write_readme(summary, missing_data, correlation_matrix, outliers, output_dir)

    print("Workflow completed successfully!")  # Debug


# -----------------------------------------------------------
# Entry point for the script
# -----------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

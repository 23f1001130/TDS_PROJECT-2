# /// Script metadata
# python-version-required = ">=3.9"
# library-dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "openai",
#   "requests",
#   "scikit-learn",
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
import openai

# Function to process and analyze data, checking its overall structure and properties
def analyze_dataset(dataset):
    print("Starting analysis of the dataset.")
    stats_summary = dataset.describe(include='all')  # Summary for numerical and categorical columns
    null_counts = dataset.isnull().sum()  # Calculate missing data for each column
    if dataset.empty:
        print("Warning: Dataset is empty.")
        correlation_data = pd.DataFrame()
    else:
        correlation_data = dataset.corr()  # Correlation matrix for numeric columns
    print("Dataset analysis completed.")
    return stats_summary, null_counts, correlation_data


# Function to detect potential outliers in numerical columns using IQR
def detect_outliers_iqr(dataframe):
    print("Detecting outliers using interquartile range (IQR)...")
    numerical_data = dataframe.select_dtypes(include=[np.number])
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR_values = Q3 - Q1
    outlier_counts = ((numerical_data < (Q1 - 1.5 * IQR_values)) | (numerical_data > (Q3 + 1.5 * IQR_values))).sum()
    print("Outlier detection completed.")
    return outlier_counts


# Function to create various plots for better data understanding
def generate_plots(correlations, outliers_data, dataframe, output_dir):
    print("Generating plots for the dataset...")
    # Create a correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm")
    heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()

    # Outlier bar chart (if applicable)
    if not outliers_data.empty and outliers_data.sum() > 0:
        outliers_data.plot(kind="bar", color="red", title="Outliers per Column")
        outliers_file = os.path.join(output_dir, "outlier_chart.png")
        plt.savefig(outliers_file)
        plt.close()
    else:
        outliers_file = None

    # Histogram of the first numerical column
    numerical_columns = dataframe.select_dtypes(include=[np.number])
    if not numerical_columns.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(numerical_columns.iloc[:, 0], kde=True, bins=25)
        histogram_file = os.path.join(output_dir, "numeric_column_histogram.png")
        plt.savefig(histogram_file)
        plt.close()
    else:
        histogram_file = None

    print("Plot generation completed.")
    return heatmap_file, outliers_file, histogram_file


# Function to create a detailed Markdown report summarizing the dataset
def create_markdown_report(summary, null_values, corr_matrix, outlier_data, folder_path):
    print("Compiling Markdown report...")
    report_path = os.path.join(folder_path, "dataset_summary_report.md")
    with open(report_path, "w") as md_report:
        md_report.write("# Dataset Analysis Report\n")
        md_report.write("## Overview\nThis report summarizes the dataset's properties and insights.\n\n")

        # Adding descriptive statistics
        md_report.write("### Descriptive Statistics\n")
        md_report.write(summary.to_markdown() + "\n\n")

        # Adding information about missing data
        md_report.write("### Missing Data Overview\n")
        md_report.write(null_values.to_markdown() + "\n\n")

        # Adding a correlation heatmap
        md_report.write("### Correlation Heatmap\n")
        md_report.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")

        # Adding outlier counts
        md_report.write("### Outlier Information\n")
        if outlier_data.sum() > 0:
            md_report.write(outlier_data.to_markdown() + "\n")
            md_report.write("![Outlier Chart](outlier_chart.png)\n\n")
        else:
            md_report.write("No outliers detected.\n\n")

        # Histogram visualization
        md_report.write("### Numeric Column Distribution\n")
        md_report.write("![Histogram](numeric_column_histogram.png)\n\n")

    print(f"Report successfully saved at {report_path}.")
    return report_path


# AI-enhanced insights generation using the OpenAI GPT API
def generate_ai_insights(prompt_text, dataset_description):
    print("Requesting AI-generated insights...")
    openai_api_key = os.getenv("AIPROXY_TOKEN")
    if not openai_api_key:
        print("API key not found. Skipping insights generation.")
        return "No insights generated due to missing API key."

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    ai_api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis expert."},
            {"role": "user", "content": f"{prompt_text}\nContext:\n{dataset_description}"}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(ai_api_url, headers=headers, json=payload)
        if response.status_code == 200:
            ai_response = response.json()
            return ai_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return "Failed to retrieve AI insights."
    except Exception as error_message:
        print(f"Error during API call: {error_message}")
        return "AI insights could not be generated."


# Entry point for the script
def main(filepath):
    print("Initializing dataset processing workflow...")

    try:
        # Load the dataset from the specified file
        data = pd.read_csv(filepath)
        print("Dataset successfully loaded.")
    except Exception as file_error:
        print(f"Error loading file: {file_error}")
        return

    # Create an output folder for all results
    results_dir = "./analysis_output"
    os.makedirs(results_dir, exist_ok=True)

    # Perform data analysis
    statistics, missing_values, correlations = analyze_dataset(data)
    outliers_detected = detect_outliers_iqr(data)

    # Generate visual plots
    generate_plots(correlations, outliers_detected, data, results_dir)

    # Generate AI-driven insights
    ai_report = generate_ai_insights("Identify trends in this dataset.", str(statistics))

    # Create a Markdown-based summary report
    report_file_path = create_markdown_report(statistics, missing_values, correlations, outliers_detected, results_dir)
    with open(report_file_path, "a") as report_append:
        report_append.write("## AI Insights\n")
        report_append.write(ai_report)

    print("Dataset processing completed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_data_file>")
    else:
        main(sys.argv[1])

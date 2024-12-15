
# /// script
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
import openai  # Ensure 'openai' package is installed for this to work

# Function to analyze data for basic statistics, null values, and correlations
def analyze_data(df):
    print("Analyzing the dataset...")
    summary_stats = df.describe()  # Descriptive stats for numerical columns
    missing_data = df.isnull().sum()  # Count of missing values per column
    correlation_matrix = df.corr() if not df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_data, correlation_matrix


# Function to identify outliers based on the IQR method
def detect_outliers(df):
    print("Detecting outliers using IQR...")
    numeric_columns = df.select_dtypes(include=[np.number])
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).sum()
    print("Outlier detection complete.")
    return outliers


# Function to generate visualizations for the data
def create_visualizations(correlation_matrix, outliers, df, output_folder):
    print("Creating visualizations...")
    # Heatmap of correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis")
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Bar plot for outliers
    if not outliers.empty and outliers.sum() > 0:
        outliers.plot(kind='bar', color='orange', title="Outliers Count")
        outlier_plot_path = os.path.join(output_folder, "outlier_plot.png")
        plt.savefig(outlier_plot_path)
        plt.close()
    else:
        outlier_plot_path = None

    # Histogram for first numeric column
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_cols.iloc[:, 0], kde=True, bins=30)
        histogram_path = os.path.join(output_folder, "distribution_histogram.png")
        plt.savefig(histogram_path)
        plt.close()
    else:
        histogram_path = None

    print("Visualizations saved.")
    return heatmap_path, outlier_plot_path, histogram_path


# Function to generate a report in Markdown format
def generate_report(stats, missing, correlations, outliers, output_folder):
    print("Generating report...")
    report_file = os.path.join(output_folder, "data_report.md")
    with open(report_file, "w") as report:
        report.write("# Automated Data Report\n")
        report.write("## Overview\nThis document summarizes key insights from the dataset.\n\n")

        report.write("### Statistics\n")
        report.write(stats.to_markdown() + "\n\n")

        report.write("### Missing Values\n")
        report.write(missing.to_markdown() + "\n\n")

        report.write("### Correlation Matrix\n")
        report.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")

        report.write("### Outliers\n")
        if outliers.sum() > 0:
            report.write(outliers.to_markdown() + "\n")
            report.write("![Outlier Plot](outlier_plot.png)\n\n")
        else:
            report.write("No significant outliers detected.\n\n")

        report.write("### Distribution\n")
        report.write("![Distribution Histogram](distribution_histogram.png)\n\n")

    print(f"Report saved at {report_file}")
    return report_file


# Function to generate insights using AI
def generate_insights(prompt, data_context):
    print("Generating insights using AI...")
    api_token = os.getenv("AIPROXY_TOKEN")
    if not api_token:
        print("Error: API Token not found.")
        return "Failed to generate insights. No API token provided."

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an analytical assistant."},
            {"role": "user", "content": f"{prompt}\nContext:\n{data_context}"}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Unable to generate insights."
    except Exception as error:
        print(f"Exception occurred: {error}")
        return "Failed to generate insights."


# Main entry point
def main(file_path):
    print("Starting the data processing pipeline...")

    # Load dataset
    try:
        dataset = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except Exception as load_error:
        print(f"Error loading file: {load_error}")
        return

    # Create output folder
    output_directory = "./output"
    os.makedirs(output_directory, exist_ok=True)

    # Perform analysis
    stats, missing, correlations = analyze_data(dataset)
    outliers = detect_outliers(dataset)

    # Create visualizations
    create_visualizations(correlations, outliers, dataset, output_directory)

    # Generate AI-powered insights
    insights = generate_insights("Analyze this dataset for key trends.", str(stats))

    # Generate and finalize report
    report_path = generate_report(stats, missing, correlations, outliers, output_directory)
    with open(report_path, "a") as report:
        report.write("## AI-Generated Insights\n")
        report.write(insights)

    print("Data processing pipeline completed successfully.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_dataset>")
    else:
        main(sys.argv[1])


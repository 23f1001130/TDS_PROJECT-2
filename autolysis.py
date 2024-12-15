# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "os",
#   "matplotlib",
#    "argparse",
#   "pandas",
#   "requests",
#   "openai",
#   "numpy",
#   "jason",
#   "ipykernel",
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
import argparse

# Function: Perform statistical analysis on a dataset
def summarize_dataset(data):
    print("Performing dataset analysis...")
    stats_summary = data.describe()
    missing_data = data.isnull().sum()
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()
    print("Dataset analysis completed.")
    return stats_summary, missing_data, correlation_matrix

# Function: Identify outliers using the Interquartile Range (IQR) method
def identify_outliers(data):
    print("Identifying outliers...")
    numeric_data = data.select_dtypes(include=[np.number])
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))).sum()
    print("Outlier identification completed.")
    return outlier_counts

# Function: Generate visual outputs from dataset analysis results
def generate_visuals(correlation_matrix, outlier_data, dataset, output_folder):
    print("Creating visualizations...")

    # Correlation matrix heatmap
    if not correlation_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        heatmap_path = os.path.join(output_folder, "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
    else:
        heatmap_path = None

    # Outlier visualization
    if not outlier_data.empty and outlier_data.sum() > 0:
        plt.figure(figsize=(10, 6))
        outlier_data.plot(kind="bar", color="red")
        plt.title("Outlier Counts by Column")
        plt.xlabel("Columns")
        plt.ylabel("Count")
        outlier_plot_path = os.path.join(output_folder, "outliers.png")
        plt.savefig(outlier_plot_path)
        plt.close()
    else:
        outlier_plot_path = None

    # Distribution plot for the first numeric column
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_col = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(dataset[first_col], kde=True, color="blue", bins=30)
        plt.title(f"Distribution of {first_col}")
        distribution_path = os.path.join(output_folder, "distribution.png")
        plt.savefig(distribution_path)
        plt.close()
    else:
        distribution_path = None

    print("Visualization generation completed.")
    return heatmap_path, outlier_plot_path, distribution_path

# Function: Create a detailed Markdown report

def create_report(stats_summary, missing_data, correlation_matrix, outliers, output_folder):
    print("Compiling report...")
    report_path = os.path.join(output_folder, "Analysis_Report.md")
    try:
        with open(report_path, "w") as report:
            report.write("# Data Analysis Report\n\n")
            report.write("## Summary Statistics\n")
            report.write(stats_summary.to_markdown() + "\n\n")

            report.write("## Missing Data\n")
            report.write(missing_data.to_markdown() + "\n\n")

            report.write("## Correlation Matrix\n")
            if not correlation_matrix.empty:
                report.write("![Correlation Matrix](heatmap.png)\n\n")
            else:
                report.write("No correlation matrix available.\n\n")

            report.write("## Outliers\n")
            report.write(outliers.to_markdown() + "\n\n")
            report.write("![Outliers Visualization](outliers.png)\n\n")

            report.write("## Data Distribution\n")
            report.write("![Data Distribution](distribution.png)\n\n")

        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Error writing report: {e}")

# Function: Query a language model for insights based on the dataset analysis
def query_language_model(prompt, analysis_context):
    print("Requesting narrative generation from LLM...")
    api_key = os.getenv("AIPROXY_TOKEN")
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{analysis_context}"}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(request_payload))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error in LLM response: {response.status_code}, {response.text}")
            return "Narrative generation failed."
    except Exception as e:
        print(f"Exception during LLM request: {e}")
        return "Narrative generation failed."

# Main script execution
def main(file_path):
    print("Starting analysis pipeline...")

    try:
        dataset = pd.read_csv(file_path, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    stats, missing, corr_matrix = summarize_dataset(dataset)
    outliers = identify_outliers(dataset)

    output_directory = "./output"
    os.makedirs(output_directory, exist_ok=True)

    heatmap, outlier_chart, dist_chart = generate_visuals(corr_matrix, outliers, dataset, output_directory)

    story = query_language_model("Generate a creative summary based on the analysis.", f"Stats: {stats}\nMissing: {missing}")

    create_report(stats, missing, corr_matrix, outliers, output_directory)

    story_path = os.path.join(output_directory, "Data_Narrative.txt")
    with open(story_path, "w") as narrative_file:
        narrative_file.write(story)

    print("Analysis pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Data Analysis")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.csv_file)

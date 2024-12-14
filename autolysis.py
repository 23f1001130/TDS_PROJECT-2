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

# Function for summarizing data with basic stats, null checks, and correlations
def summarize_dataset(dataframe):
    print("Starting dataset summary...")  # Debug
    # Generate descriptive statistics for numerical data
    statistics = dataframe.describe()

    # Count missing values in each column
    missing_counts = dataframe.isna().sum()

    # Correlation analysis for numerical features
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation = numeric_columns.corr() if not numeric_columns.empty else pd.DataFrame()

    print("Dataset summary complete.")  # Debug
    return statistics, missing_counts, correlation


# Function to identify outliers using Interquartile Range (IQR)
def find_outliers(dataframe):
    print("Identifying outliers...")  # Debug
    # Focus on numeric columns
    numeric_data = dataframe.select_dtypes(include=[np.number])

    # Calculate the IQR and detect outliers
    lower_quartile = numeric_data.quantile(0.25)
    upper_quartile = numeric_data.quantile(0.75)
    iqr = upper_quartile - lower_quartile
    outliers = ((numeric_data < (lower_quartile - 1.5 * iqr)) | 
                (numeric_data > (upper_quartile + 1.5 * iqr))).sum()

    print("Outlier identification complete.")  # Debug
    return outliers


# Function to create visual outputs like heatmaps, outlier plots, and distributions
def create_visuals(correlation_matrix, outlier_counts, dataframe, save_dir):
    print("Creating visual outputs...")  # Debug
    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    heatmap_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Generate bar chart for outliers if any exist
    if not outlier_counts.empty and outlier_counts.sum() > 0:
        plt.figure(figsize=(10, 6))
        outlier_counts.plot(kind='bar', color='orange')
        plt.title('Detected Outliers')
        plt.xlabel('Features')
        plt.ylabel('Outlier Count')
        outliers_path = os.path.join(save_dir, 'outlier_plot.png')
        plt.savefig(outliers_path)
        plt.close()
    else:
        print("No significant outliers detected.")
        outliers_path = None

    # Plot distribution of the first numeric column
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_cols.any():
        first_col = numeric_cols[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[first_col], kde=True, bins=30, color='teal')
        plt.title(f'Distribution of {first_col}')
        dist_path = os.path.join(save_dir, 'distribution_plot.png')
        plt.savefig(dist_path)
        plt.close()
    else:
        dist_path = None

    print("Visual outputs complete.")  # Debug
    return heatmap_path, outliers_path, dist_path


# Function to compile findings and visuals into a markdown report
def generate_report(stats, missing_data, correlation, outliers, output_dir):
    print("Generating report...")  # Debug
    report_path = os.path.join(output_dir, 'REPORT.md')
    try:
        with open(report_path, 'w') as report:
            report.write("# Automated Data Analysis Report\n\n")
            report.write("## Summary Statistics\n")
            report.write(f"Dataset Overview:\n\n{stats.to_markdown()}\n\n")
            report.write("## Missing Data\n")
            for col, val in missing_data.items():
                report.write(f"- {col}: {val} missing values\n")
            report.write("\n")

            report.write("## Outliers\n")
            for feature, count in outliers.items():
                report.write(f"- {feature}: {count} detected outliers\n")

            report.write("## Correlation Heatmap\n")
            report.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
            report.write("## Outlier Chart\n")
            if outliers_path:
                report.write("![Outliers](outlier_plot.png)\n\n")
            else:
                report.write("No outliers to display.\n\n")

            report.write("## Feature Distribution\n")
            if dist_path:
                report.write("![Distribution](distribution_plot.png)\n\n")
            else:
                report.write("No numeric columns to plot.\n\n")

        print(f"Report saved: {report_path}")  # Debug
        return report_path
    except Exception as e:
        print(f"Error writing report: {e}")
        return None


# Function to leverage OpenAI's API for generating narratives
def use_ai_story(prompt, context):
    print("Requesting story generation...")  # Debug
    try:
        token = os.environ.get("AIPROXY_TOKEN", "")
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an insightful writer."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error in AI request: {response.status_code}")
            return "Failed to retrieve story."
    except Exception as e:
        print(f"Exception during AI request: {e}")
        return "Error occurred."


# Main function tying everything together
def main(csv_path):
    print("Starting data analysis pipeline...")  # Debug
    try:
        data = pd.read_csv(csv_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    stats, missing_data, correlation = summarize_dataset(data)
    outliers = find_outliers(data)

    output_directory = "./analysis_results"
    os.makedirs(output_directory, exist_ok=True)

    heatmap_path, outliers_path, dist_path = create_visuals(correlation, outliers, data, output_directory)

    story_prompt = "Generate a narrative based on the provided data insights."
    story_context = f"Stats:\n{stats}\n\nMissing:\n{missing_data}\nOutliers:\n{outliers}"
    story = use_ai_story(story_prompt, story_context)

    report_path = generate_report(stats, missing_data, correlation, outliers, output_directory)
    if report_path:
        with open(report_path, 'a') as report:
            report.write("\n## Generated Story\n")
            report.write(story)

    print("Analysis complete. All files saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <csv_file_path>")
    else:
        main(sys.argv[1])

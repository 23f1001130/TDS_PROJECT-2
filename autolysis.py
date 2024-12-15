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
import argparse
import requests
import json
import openai  # Ensure this package is installed: pip install openai

# Perform data analysis: statistics, missing values, and correlations
def analyze_data(df):
    print("Starting data examination...")  # Debugging prompt
    stats_summary = df.describe()  # Numerical data summary
    null_counts = df.isnull().sum()  # Missing values tally
    numeric_cols = df.select_dtypes(include=[np.number])  # Filter numerical columns
    correlations = numeric_cols.corr() if not numeric_cols.empty else pd.DataFrame()
    print("Data analysis completed.")  # Debugging prompt
    return stats_summary, null_counts, correlations


# Identify outliers using the IQR approach
def detect_outliers(df):
    print("Initiating outlier detection...")  # Debugging prompt
    numeric_data = df.select_dtypes(include=[np.number])  # Numeric columns
    Q1, Q3 = numeric_data.quantile(0.25), numeric_data.quantile(0.75)
    iqr = Q3 - Q1
    outlier_count = ((numeric_data < (Q1 - 1.5 * iqr)) | (numeric_data > (Q3 + 1.5 * iqr))).sum()
    print("Outlier identification completed.")  # Debugging prompt
    return outlier_count


# Generate visualizations: heatmap, outlier plots, distribution plots
def visualize_data(corr_matrix, outliers, df, output_dir):
    print("Creating data visualizations...")  # Debugging prompt

    # Heatmap for correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    corr_file = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_file)
    plt.close()

    # Plot for outliers (if any exist)
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(6, 6))
        outliers.plot(kind="bar", color="red")
        plt.title("Outlier Detection Visualization")
        plt.xlabel("Columns")
        plt.ylabel("Count of Outliers")
        outliers_file = os.path.join(output_dir, "outliers.png")
        plt.savefig(outliers_file)
        plt.close()
    else:
        outliers_file = None

    # Distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(6, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, bins=30, color="blue")
        plt.title("Data Distribution")
        dist_file = os.path.join(output_dir, "distribution_plot.png")
        plt.savefig(dist_file)
        plt.close()
    else:
        dist_file = None

    print("Visualizations created successfully.")  # Debugging prompt
    return corr_file, outliers_file, dist_file


# Generate a report (README.md) containing analysis results
def generate_report(stats_summary, missing_values, corr_matrix, outliers, output_dir):
    print("Drafting analysis report...")  # Debugging prompt
    readme_path = os.path.join(output_dir, "README.md")
    try:
        with open(readme_path, "w") as file:
            file.write("# Automated Data Analysis Report\n\n")
            file.write("## Summary\n")
            file.write("This report includes an automated overview of the dataset, along with visualizations and key insights.\n\n")

            # Adding Summary Stats
            file.write("## Summary Statistics\n")
            file.write("| Metric    | Value |\n|-----------|-------|\n")
            for col in stats_summary.columns:
                file.write(f"| {col} - Mean | {stats_summary.loc['mean', col]:.2f} |\n")

            # Missing Values Section
            file.write("## Missing Data\n")
            file.write("| Column    | Count |\n|-----------|-------|\n")
            for col, count in missing_values.items():
                file.write(f"| {col} | {count} |\n")

            # Outlier Counts
            file.write("## Outliers\n")
            file.write("| Column    | Outliers |\n|-----------|----------|\n")
            for col, count in outliers.items():
                file.write(f"| {col} | {count} |\n")

            # Correlation Matrix Visualization
            file.write("## Correlation Heatmap\n")
            file.write("![Heatmap](correlation_heatmap.png)\n\n")

            # Outliers Visualization
            if outliers.sum() > 0:
                file.write("## Outlier Visualization\n")
                file.write("![Outliers](outliers.png)\n\n")

            # Distribution Plot
            file.write("## Distribution of Data\n")
            file.write("![Distribution](distribution_plot.png)\n\n")

            file.write("## Conclusion\nThe dataset was analyzed for patterns, relationships, and anomalies.\n\n")
        print(f"Report saved at: {readme_path}")  # Debugging prompt
        return readme_path
    except Exception as e:
        print(f"Error in report generation: {e}")
        return None


# Generate narrative insights using an OpenAI model via a proxy
def generate_story(prompt, context):
    try:
        token = os.environ.get("AIPROXY_TOKEN", "")
        if not token:
            raise ValueError("Missing API Token")

        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a narrative assistant."},
                {"role": "user", "content": f"{prompt}\nContext:\n{context}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Unable to generate story."
    except Exception as e:
        print(f"Exception: {e}")
        return "Failed to generate story."


# Main function to run the analysis
def main(file_path):
    print("Initializing process...")  # Debugging prompt
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        print("Dataset loaded successfully!")  # Debugging prompt
    except Exception as e:
        print(f"File read error: {e}")
        return

    stats, missing, corr = analyze_data(df)
    outliers = detect_outliers(df)
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    visualize_data(corr, outliers, df, output_dir)
    story = generate_story("Analyze the dataset creatively.", context=str(stats))

    report_path = generate_report(stats, missing, corr, outliers, output_dir)
    if report_path:
        with open(report_path, "a") as f:
            f.write("\n## Narrative Analysis\n")
            f.write(story)
        print(f"Process completed. Report: {report_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
    else:
        main(sys.argv[1])

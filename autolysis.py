# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "matplotlib",
#   "pandas",
#   "requests",
#   "openai",
#   "chardet"  # Add all packages used in the script
# ]
# ///
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")
# Function to create directories if they do not exist
def ensure_directories_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        with open(os.path.join(directory, "README.md"), "w") as f:
            f.write(f"# {directory.capitalize()} Analysis\n\nPlaceholder README for {directory} analysis.")

# Function to handle missing values and outliers
def preprocess_data(df):
    # Handle missing values
    missing_summary = df.isnull().sum()
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill numeric NaNs with mean

    # Detect and remove outliers using Z-scores
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    df_cleaned = df[(z_scores < 3).all(axis=1)]

    return df_cleaned, missing_summary

# Function to generate visualizations
def generate_visualizations(df, output_dir):
    # Select the numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns

    # 1. Histogram of a key numerical feature (choose the first numeric column)
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_cols[0]], kde=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "histogram.png"))
        plt.close()

    # 2. Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # 3. Pairplot of the top 3 numeric features
    if len(numeric_cols) > 2:
        sns.pairplot(df[numeric_cols[:3]])  # Use the first 3 numeric columns for the pairplot
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()

# Function to generate Markdown narratives
def generate_narrative(df, missing_summary, output_dir):
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    corr_matrix = df[numeric_cols].corr()

    insights = []
    insights.append("# Dataset Analysis\n")
    insights.append("## Summary Statistics\n")
    insights.append(df.describe().to_markdown())
    insights.append("\n\n## Missing Values\n")
    insights.append(missing_summary.to_markdown())
    insights.append("\n\n## Correlation Matrix\n")
    insights.append(corr_matrix.to_markdown())

    insights.append("\n\n## Key Insights\n")
    for col in numeric_cols:
        insights.append(f"- *{col}:* Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}, Skewness = {df[col].skew():.2f}, Kurtosis = {df[col].kurt():.2f}")

    insights.append("\n\n## Visualizations\n")
    insights.append("Refer to the generated PNG files for detailed visualizations, including distribution histograms and correlation heatmaps.")

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("\n".join(insights))

# Main function to process datasets
def process_dataset(dataset_path, output_dir):
    try:
        # Read the dataset
        df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        # Clean the data
        df_cleaned, missing_summary = preprocess_data(df)
        # Generate the visualizations
        generate_visualizations(df_cleaned, output_dir)
        # Generate the narrative
        generate_narrative(df_cleaned, missing_summary, output_dir)
    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")

# Script entry point
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Analyze datasets and generate narratives and visualizations.")
    parser.add_argument("dataset", help="Path to the input dataset (CSV file).")
    args = parser.parse_args()

    # Define output directory dynamically based on dataset name (use the base name of the file)
    dataset_name = os.path.basename(args.dataset).split('.')[0].lower()
    output_dir = dataset_name

    # Ensure the output directory exists
    ensure_directories_exist(output_dir)

    # Process the dataset
    process_dataset(args.dataset, output_dir)


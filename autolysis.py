
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
#   "tabulate",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("analysis.log"), logging.StreamHandler()]
)

# Function to load a dataset from a local file or URL
def load_csv(file_path):
    try:
        if file_path.startswith("http"):
            logging.info("Loading remote CSV file.")
            df = pd.read_csv(file_path)
        else:
            logging.info("Loading local CSV file.")
            df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Function to summarize the dataset
def summarize_dataset(df):
    try:
        logging.info("Generating dataset summary.")
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        logging.info("Summary generated successfully.")
        return summary
    except Exception as e:
        logging.error(f"Error summarizing dataset: {e}")
        raise

# Function to detect anomalies in numerical columns
def detect_anomalies(df):
    try:
        logging.info("Detecting anomalies using IQR method.")
        anomalies = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()
        logging.info("Anomaly detection completed.")
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        raise

# Function to generate visualizations
def generate_visualizations(df, output_dir):
    try:
        logging.info("Generating visualizations.")
        os.makedirs(output_dir, exist_ok=True)

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        logging.info(f"Correlation heatmap saved to {heatmap_path}.")

        # Distribution plots
        dist_plots = []
        for col in df.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True)
            dist_path = os.path.join(output_dir, f"{col}_distribution.png")
            plt.savefig(dist_path)
            plt.close()
            dist_plots.append(dist_path)
            logging.info(f"Distribution plot saved for {col}.")

        return {"heatmap": heatmap_path, "distributions": dist_plots}
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        raise

# Function to generate a Markdown report
def generate_report(summary, anomalies, viz_paths, output_dir):
    try:
        logging.info("Generating Markdown report.")
        report_path = os.path.join(output_dir, "ANALYSIS_REPORT.md")
        with open(report_path, "w") as f:
            f.write("# Dataset Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            f.write("## Dataset Summary\n\n")
            for key, value in summary.items():
                f.write(f"- **{key.capitalize()}**: {value}\n")
            f.write("\n")

            # Anomalies
            f.write("## Anomalies\n\n")
            for col, values in anomalies.items():
                f.write(f"- **{col}**: {len(values)} anomalies detected\n")
            f.write("\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write(f"![Correlation Heatmap]({viz_paths['heatmap']})\n\n")
            for dist in viz_paths['distributions']:
                f.write(f"![Distribution Plot]({dist})\n")

        logging.info(f"Report saved to {report_path}.")
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise

# Main function
def main():
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis Tool.")
    parser.add_argument("file", help="Path to the dataset (CSV file) or remote URL.")
    parser.add_argument("--output", default="results", help="Directory to save results.")
    args = parser.parse_args()

    try:
        df = load_csv(args.file)
        summary = summarize_dataset(df)
        anomalies = detect_anomalies(df)
        viz_paths = generate_visualizations(df, args.output)
        generate_report(summary, anomalies, viz_paths, args.output)
        logging.info("Dataset analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in dataset analysis: {e}")

if __name__ == "__main__":
    main()

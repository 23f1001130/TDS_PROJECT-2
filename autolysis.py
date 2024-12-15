
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
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress Matplotlib font manager debug messages
import matplotlib
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING) 
# --- Helper Functions ---
def compute_summary_statistics(df):
    """Generate descriptive statistics for the dataset."""
    return df.describe(include='all').fillna('N/A')

def detect_outliers(df, col):
    """Detect outliers using the Interquartile Range (IQR) method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    return outliers if not outliers.empty else pd.DataFrame({'Message': ['No outliers detected']})

def plot_correlation_matrix(df, output_path):
    """Generate and save a correlation matrix heatmap."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(output_path)
    plt.close()

def generate_narrative(summary_stats, outliers, dataset_name):
    """Generate a narrative report for the dataset."""
    narrative = f"# Report for {dataset_name}\n\n"
    narrative += "## Dataset Overview\n"
    narrative += f"Number of rows: {summary_stats.shape[0]}\n"
    narrative += f"Number of columns: {summary_stats.shape[1]}\n\n"
    narrative += "## Summary Statistics\n" + summary_stats.to_markdown() + "\n\n"
    if not outliers.empty:
        narrative += "## Outliers Detected\n" + outliers.to_markdown() + "\n\n"
    else:
        narrative += "## Outliers Detected\nNo significant outliers found.\n\n"
    narrative += "## Visualizations\n"
    narrative += "- Correlation Heatmap\n"
    narrative += "- Pair Plot\n"
    narrative += "- Boxplot\n"
    return narrative

def generate_visualizations(df, dataset_name, output_dir):
    """Generate various visualizations for the dataset."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Pair plot
    pairplot_path = os.path.join(output_dir, f"{dataset_name}_pairplot.png")
    sns.pairplot(df[numeric_columns])
    plt.savefig(pairplot_path)
    plt.close()

    # Boxplot for outliers
    boxplot_path = os.path.join(output_dir, f"{dataset_name}_boxplot.png")
    df[numeric_columns].plot(kind='box', figsize=(6, 6))
    plt.title("Boxplot of Numeric Features")
    plt.savefig(boxplot_path)
    plt.close()

    # Bar chart for categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if df[col].nunique() < 20:  # Limit to avoid overly dense plots
            plt.figure(figsize=(8, 6))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f"Frequency of {col}")
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{col}_barplot.png"))
            plt.close()

def validate_metadata(output_dir):
    """Ensure required files are present."""
    required_files = ["README.md", "correlation.png", "pairplot.png", "boxplot.png"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            missing_files.append(file)
    if missing_files:
        logging.warning(f"Missing required files: {', '.join(missing_files)}")
    else:
        logging.info("All required files are present.")

def check_visual_output(output_dir):
    """Verify visual outputs."""
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            img = Image.open(os.path.join(output_dir, file))
            logging.info(f"{file} - Dimensions: {img.size}, Mode: {img.mode}")

# --- Main Processing Function ---
import chardet

def process_dataset(file_path, output_dir):
    """Process a dataset: analyze, visualize, and save outputs."""
    try:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Detect and handle file encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        logging.info(f"Detected encoding for {file_path}: {encoding}")

        # Load the dataset with detected encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        narrative_path = os.path.join(output_dir, f"{dataset_name}_report.md")
        visualization_path = os.path.join(output_dir, f"{dataset_name}_correlation.png")

        # Perform analysis
        summary_stats = compute_summary_statistics(df)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = pd.concat([detect_outliers(df, col) for col in numeric_columns])

        # Generate visualizations
        if not df[numeric_columns].empty:
            plot_correlation_matrix(df[numeric_columns], visualization_path)
            generate_visualizations(df, dataset_name, output_dir)

        # Generate narrative
        narrative = generate_narrative(summary_stats, outliers, dataset_name)

        # Save narrative
        with open(narrative_path, "w", encoding="utf-8") as f:
            f.write(narrative)

        logging.info(f"Analysis complete for {dataset_name}. Outputs saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")


# --- Main Function ---
def main(input_path, output_dir="results"):
    """Main entry point for analysis."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith(".csv"):
                    process_dataset(os.path.join(input_path, file), output_dir)
        else:
            process_dataset(input_path, output_dir)

        validate_metadata(output_dir)
        check_visual_output(output_dir)
    except Exception as e:
        logging.error(f"Error in main processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autolysis script for CSV analysis.")
    parser.add_argument("input_path", help="Path to the input CSV file or directory.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results",  # Default value for output_dir
        help="Path to the output directory (default: 'results')."
    )
    args = parser.parse_args()
    main(args.input_path, args.output_dir)

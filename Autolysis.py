import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.express as px
import chardet
import logging

logging.basicConfig(filename='errors.log', level=logging.ERROR)

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def compute_summary_statistics(df):
    """Generate descriptive statistics for the dataset."""
    stats = df.describe(include='all').fillna('N/A')
    additional_stats = {
        'Skewness': df.skew(numeric_only=True).to_dict(),
        'Kurtosis': df.kurt(numeric_only=True).to_dict()
    }
    return stats, additional_stats

def detect_outliers(df, column):
    """Detect outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def generate_visualizations(df, output_path):
    """Generate and save visualizations."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if numeric_columns.empty:
        print("No numeric columns found. Skipping visualizations.")
        return

    try:
        # Correlation heatmap
        correlation_matrix = df[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(output_path / 'correlation_heatmap.png')
        plt.close()

        # Pair plot (limited for efficiency)
        high_variance_cols = df[numeric_columns].var().nlargest(5).index
        sns.pairplot(df[high_variance_cols])
        plt.savefig(output_path / 'pairplot.png')
        plt.close()

        # Interactive visualization using Plotly
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis')
        fig.write_html(str(output_path / 'correlation_heatmap_interactive.html'))
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        logging.error(f"Visualization generation failed: {e}")

def generate_report(df, output_path):
    """Generate a markdown report for the dataset."""
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Compute statistics
        stats, additional_stats = compute_summary_statistics(df)

        # Write report
        report_path = output_path / 'report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Analysis Report\n\n")
            f.write("## Summary Statistics\n")
            f.write(stats.to_markdown() + "\n\n")

            f.write("## Additional Statistics\n")
            for key, value in additional_stats.items():
                f.write(f"### {key}\n")
                f.write(pd.DataFrame(value, index=[0]).to_markdown() + "\n\n")

            # Outlier detection
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            outliers_list = [detect_outliers(df, col) for col in numeric_columns]
            outliers = pd.concat(outliers_list) if any(not o.empty for o in outliers_list) else pd.DataFrame()
            if not outliers.empty:
                f.write("## Outliers\n")
                f.write(outliers.to_markdown() + "\n\n")
            else:
                f.write("## Outliers\nNo outliers detected.\n\n")

            f.write("[Interactive Correlation Heatmap](correlation_heatmap_interactive.html)\n\n")

        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Report generation failed: {e}")
        logging.error(f"Report generation failed: {e}")

def main(input_file):
    """Main function to analyze the dataset."""
    output_path = Path("output") / Path(input_file).stem
    try:
        encoding = detect_encoding(input_file)
        df = pd.read_csv(input_file, encoding=encoding)
        print(f"File '{input_file}' loaded successfully with encoding '{encoding}'.")
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        logging.error(f"Error reading file {input_file}: {e}")
        return

    # Handle missing data
    missing_data = df.isnull().sum() / len(df) * 100
    if missing_data.max() > 50:
        print(f"Dataset '{input_file}' contains excessive missing data. Cleaning...")
        df = df.dropna(axis=1, thresh=len(df) * 0.5)

    try:
        generate_report(df, output_path)
    except Exception as e:
        print(f"Report generation failed: {e}")
        logging.error(f"Report generation failed for {input_file}: {e}")

    try:
        generate_visualizations(df, output_path)
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        logging.error(f"Visualization generation failed for {input_file}: {e}")

    print(f"Analysis completed successfully for {input_file}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Analysis Tool")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    main(args.input_file)

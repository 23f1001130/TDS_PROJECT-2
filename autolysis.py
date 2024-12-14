# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "openai",
#   "tabulate",
#   "scikit-learn",
#   "requests",
#   "ipykernel",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy"
# ]
# ///
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.express as px

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

    # Correlation heatmap
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(output_path / 'correlation_heatmap.png')
    plt.close()

    # Pair plot (limited for efficiency)
    if len(numeric_columns) > 5:
        sns.pairplot(df[numeric_columns[:5]])
    else:
        sns.pairplot(df[numeric_columns])
    plt.savefig(output_path / 'pairplot.png')
    plt.close()

    # Interactive visualization using Plotly
    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis')  # Changed 'coolwarm' to 'Viridis'
    fig.write_html(str(output_path / 'correlation_heatmap_interactive.html'))


def generate_report(df, output_path):
    """Generate a markdown report for the dataset."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    stats, additional_stats = compute_summary_statistics(df)

    # Write report
    report_path = output_path / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
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

    print(f"Report generated: {report_path}")


def main(input_file):
    """Main function to analyze the dataset."""
    output_path = Path("output") / Path(input_file).stem
    try:
        df = pd.read_csv(input_file, encoding='utf-8')  # Attempt to read with UTF-8
    except UnicodeDecodeError:
        print(f"UTF-8 encoding failed for {input_file}. Retrying with 'latin1' encoding...")
        try:
            df = pd.read_csv(input_file, encoding='latin1')  # Fallback to latin1
        except Exception as e:
            print(f"Error reading file {input_file} with fallback encoding: {e}")
            return

    # Handle datasets with excessive missing data
    missing_data = df.isnull().sum() / len(df) * 100
    if missing_data.max() > 50:
        print(f"Dataset '{input_file}' contains too many missing values. Consider cleaning.")
        return

    generate_report(df, output_path)
    generate_visualizations(df, output_path)
    print(f"Analysis completed successfully for {input_file}.")


if __name__ == "__main__":
    # Example usage
    input_files = ["goodreads.csv", "happiness.csv", "media.csv"]
    for file in input_files:
        print(f"Processing {file}...")
        main(file)

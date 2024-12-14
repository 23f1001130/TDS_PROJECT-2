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
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import chardet


# Ensure the AIPROXY_TOKEN environment variable is set
api_proxy_token = os.getenv("AIPROXY_TOKEN")
if not api_proxy_token:
    raise EnvironmentError("AIPROXY_TOKEN environment variable is required.")

# Set the base API URL for the proxy
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

# Ensure a CSV file is provided as an argument
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

csv_file_path = sys.argv[1]
if not os.path.isfile(csv_file_path) or not csv_file_path.endswith(".csv"):
    print(f"Error: {csv_file_path} is not a valid CSV file.")
    sys.exit(1)


def detect_encoding(file_path):
    """Detect file encoding."""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(1024))
        return result["encoding"]


def load_csv(file_path):
    """Load CSV with flexible encoding and parse dates."""
    encodings = [detect_encoding(file_path), "utf-8", "latin1"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, parse_dates=True)
        except Exception:
            continue
    raise ValueError("Failed to load CSV with multiple encodings.")


def perform_analysis(df):
    """Perform generic analysis on the dataset."""
    # Ensure non-numeric columns are handled gracefully
    numeric_df = df.select_dtypes(include=["number"])

    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    if not numeric_df.empty:
        analysis["correlation_matrix"] = numeric_df.corr().to_dict()
    return analysis


def generate_visualizations(df, output_prefix):
    """Generate PNG visualizations from the dataset."""
    charts = []
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        # Correlation Heatmap
        plt.figure(figsize=(6,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_file = f"{output_prefix}_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_file, dpi=150)
        charts.append(heatmap_file)
        plt.close()

        # Histogram of the first numeric column
        plt.figure(figsize=(6, 6))
        df[numeric_cols[0]].hist(bins=30, color="skyblue", edgecolor="black")
        hist_file = f"{output_prefix}_{numeric_cols[0]}_histogram.png"
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.savefig(hist_file, dpi=150)
        charts.append(hist_file)
        plt.close()

    return charts


def llm_narrate(analysis, charts):
    """Use LLM to generate a narrative for the dataset analysis."""
    openai.api_key = api_proxy_token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"  # Ensure the base API is set correctly.

    # Construct the prompt for GPT-4o-Mini
    prompt = f"""
    I analyzed a dataset with the following characteristics:
    - Shape: {analysis['shape']}
    - Columns: {list(analysis['columns'].keys())}
    - Missing Values: {analysis['missing_values']}
    - Summary Statistics: {analysis['summary_statistics']}
    
    I also generated the following visualizations:
    {charts}
    
    Please provide:
    1. A brief description of the dataset.
    2. Insights from the analysis.
    3. Recommendations based on the findings.
    4. A Markdown-formatted narrative for README.md.
    """

    # Use the ChatCompletion endpoint for GPT-4o-Mini
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a highly skilled data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

def save_readme(content, charts):
    """Save the narrative to README.md and embed charts."""
    with open("README.md", "w") as f:
        f.write(content)
        f.write("\n\n## Visualizations\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")

def main():
    try:
        # Load dataset
        df = load_csv(csv_file_path)

        # Perform analysis
        analysis = perform_analysis(df)

        # Generate visualizations
        charts = generate_visualizations(df, "chart")

        # Use LLM for narrative
        narrative = llm_narrate(analysis, charts)

        # Save narrative to README.md
        save_readme(narrative, charts)

        print("Analysis complete. README.md and charts generated.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

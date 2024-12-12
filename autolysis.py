# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "matplotlib",
#   "pandas",
#   "requests",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from scipy import stats

# Retrieve API token from environment variables
from dotenv import load_dotenv
load_dotenv()

# Debugging the environment variable loading
api_proxy_token = os.environ.get("AIPROXY_TOKEN")
print(f"Debug - AIPROXY_TOKEN: {api_proxy_token}")

if not api_proxy_token:
    print("Error: API token not set in environment variables. Exiting.")
    exit(1)

# Define the API base URL
api_proxy_base_url = "https://aiproxy.sanand.workers.dev/openai/v1"

def read_csv(filename):
    """Read the CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        print(f"Dataset loaded: {filename}")
        return df
    except UnicodeDecodeError:
        print(f"Encoding issue detected with {filename}. Trying 'latin1'.")
        return pd.read_csv(filename, encoding="latin1")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        exit()

def analyze_data(df):
    """Perform basic analysis on the dataset."""
    analysis = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict()
    }
    return analysis

def visualize_data(df, output_prefix, subdirectory):
    """Generate visualizations for the dataset using Seaborn."""
    charts = []
    os.makedirs(subdirectory, exist_ok=True)

    # Example 1: Correlation Heatmap (if numeric data exists)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(14, 12))  
        heatmap = sns.heatmap(
            df[numeric_columns].corr(), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            cbar_kws={'shrink': 0.8}
        )
        heatmap.set_title("Correlation Heatmap", fontsize=12)
        heatmap_file = os.path.join(subdirectory, f"{output_prefix}_heatmap.png")
        plt.savefig(heatmap_file, dpi=300)
        charts.append(f"/{subdirectory}/{os.path.basename(heatmap_file)}")
        plt.close()

    # Example 2: Bar Plot for the first categorical column
    categorical_columns = df.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        plt.figure(figsize=(14, 8)) 
        top_categories = df[categorical_columns[0]].value_counts().head(10)
        sns.barplot(
            x=top_categories.values, 
            y=top_categories.index, 
            palette="Blues_d"
        )
        plt.title(f"Top 10 {categorical_columns[0]} Categories", fontsize=12)
        plt.xlabel("Count", fontsize=10)
        plt.ylabel(categorical_columns[0], fontsize=10)
        barplot_file = os.path.join(subdirectory, f"{output_prefix}_barplot.png")
        plt.savefig(barplot_file, dpi=300)
        charts.append(f"/{subdirectory}/{os.path.basename(barplot_file)}")
        plt.close()

    return charts

def narrate_story(analysis, charts, filename):
    """Use GPT-4o-Mini to narrate a story about the analysis."""
    summary_prompt = f"""
    I analyzed a dataset from {filename}. It has the following details:
    - Shape: {analysis['shape']}
    - Columns: {analysis['columns']}
    - Missing Values: {analysis['missing_values']}

    Write a short summary of the dataset, key insights, and recommendations. Refer to the charts where necessary.
    """
    url = f"{api_proxy_base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_proxy_token}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": summary_prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Story generation failed: {e}"

def save_markdown(story, charts, output_file):
    """Save the narrated story and chart references to a README.md file."""
    with open(output_file, "w") as f:
        f.write("# Analysis Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(story + "\n\n")
        f.write("## Visualizations\n")
        for chart in charts:
            f.write(f"![Chart]({chart})\n")

def main():
    # Accept CSV filename and folder path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        return

    dataset_filename = sys.argv[1]
    if not os.path.exists(dataset_filename):
        print(f"Error: File {dataset_filename} does not exist.")
        return

    # Determine subdirectory based on file name
    if "goodreads" in dataset_filename.lower():
        subdirectory = "goodreads"
    elif "happiness" in dataset_filename.lower():
        subdirectory = "happiness"
    else:
        subdirectory = "media"

    # Load dataset
    df = read_csv(dataset_filename)

    # Analyze dataset
    analysis = analyze_data(df)

    # Visualize data
    output_prefix = os.path.splitext(os.path.basename(dataset_filename))[0]
    charts = visualize_data(df, output_prefix, subdirectory)

    # Narrate story
    story = narrate_story(analysis, charts, dataset_filename)

    # Save README.md
    readme_file = os.path.join(subdirectory, "README.md")
    save_markdown(story, charts, readme_file)
    print(f"Analysis completed for {dataset_filename}. Check {readme_file} and charts.")

if __name__ == "__main__":
    main()

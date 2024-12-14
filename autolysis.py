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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import openai
import chardet
import sys

# Set the AIPROXY_TOKEN environment variable
os.environ["AIPROXY_TOKEN"] = "your_actual_token_here"  # Set the token directly here

# Prompt the user for their API token (if you want user input instead)
api_proxy_token = os.environ.get("AIPROXY_TOKEN", "Token not found")  # Use the token from the environment variable

if api_proxy_token == "Token not found":
    raise ValueError("API proxy token is required.")

# Ensure a CSV file is provided as a system argument
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the CSV file as a command-line argument.")

csv_file_path = sys.argv[1]
if not os.path.isfile(csv_file_path) or not csv_file_path.lower().endswith(".csv"):
    raise ValueError("A valid CSV file path is required.")

# Function to detect the encoding of a file
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(1024))  # Read the first 1 KB for detection
        return result['encoding']

# Function to read a CSV file
def read_csv(filename):
    encodings_to_try = [detect_encoding(filename), 'utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"Dataset loaded: {filename} (Encoding: {encoding})")
            return df
        except Exception as e:
            print(f"Failed with encoding {encoding}: {e}")
    print(f"All encoding attempts failed for {filename}.")
    return None

# Function to analyze the dataset
def analyze_data(df):
    try:
        analysis = {
            "shape": df.shape,
            "columns": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_statistics": df.describe(include="all").to_dict(),
        }
        numeric_data = df.select_dtypes(include=["number"])
        if not numeric_data.empty:
            analysis["correlation_matrix"] = numeric_data.corr().to_dict()
        else:
            analysis["correlation_matrix"] = None
        return analysis
    except Exception as e:
        print(f"Error analyzing data: {e}")
        traceback.print_exc()
        return {}

# Function to visualize the dataset
def visualize_data(df, output_prefix):
    charts = []
    try:
        # Correlation Heatmap
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
            heatmap.set_title("Correlation Heatmap")
            heatmap_file = f"{output_prefix}_heatmap.png"
            plt.savefig(heatmap_file, dpi=300)
            charts.append(heatmap_file)
            plt.close()

        # Distribution of numerical columns
        for column in numeric_columns:
            plt.figure(figsize=(8, 5))
            df[column].hist(bins=30, color="skyblue", edgecolor="black")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            dist_file = f"{output_prefix}_{column}_distribution.png"
            plt.savefig(dist_file, dpi=300)
            charts.append(dist_file)
            plt.close()
    except Exception as e:
        print(f"Error visualizing data: {e}")
        traceback.print_exc()
    return charts

# Function to interact with the LLM
def interact_with_llm(filename, analysis, api_token):
    try:
        openai.api_key = api_token  # Set the API key for the new interface

        prompt = (
            f"I analyzed a dataset named '{filename}' with the following details:\n"
            f"- Shape: {analysis.get('shape')}\n"
            f"- Columns and Types: {analysis.get('columns')}\n"
            f"- Missing Values: {analysis.get('missing_values')}\n"
            f"- Summary Statistics: {analysis.get('summary_statistics')}\n\n"
            "Please summarize the dataset, provide key insights, and suggest recommendations."
        )

        # Use the new completions API from openai 1.0.0
        response = openai.completions.create(
            model="gpt-4",  # Specify the model you want to use
            prompt=prompt,
            max_tokens=1000  # Adjust token length as needed
        )
        return response['choices'][0]['text'].strip()  # Extracting the result from the API response
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        traceback.print_exc()
        return "Failed to generate insights from the LLM."

# Function to save the analysis and insights to a Markdown file
def save_markdown(analysis, charts, insights, output_file):
    try:
        with open(output_file, "w") as f:
            f.write("# Analysis Report\n\n")
            f.write("## Dataset Analysis\n")
            f.write(f"Shape: {analysis.get('shape')}\n")
            f.write(f"Columns:\n{analysis.get('columns')}\n")
            f.write(f"Missing Values:\n{analysis.get('missing_values')}\n")
            f.write(f"Summary Statistics:\n{analysis.get('summary_statistics')}\n")
            f.write("\n## LLM Insights\n")
            f.write(insights + "\n")
            f.write("\n## Charts\n")
            for chart in charts:
                f.write(f"![{chart}]({chart})\n")
    except Exception as e:
        print(f"Error saving Markdown file: {e}")
        traceback.print_exc()

# Main function to process the CSV file
def main():
    print(f"Processing {csv_file_path}...")
    df = read_csv(csv_file_path)
    if df is None:
        return

    analysis = analyze_data(df)
    output_prefix = os.path.splitext(os.path.basename(csv_file_path))[0]
    charts = visualize_data(df, output_prefix)
    insights = interact_with_llm(csv_file_path, analysis, api_proxy_token)

    readme_file = f"{output_prefix}_README.md"
    save_markdown(analysis, charts, insights, readme_file)
    print(f"Completed analysis for {csv_file_path}. Results saved to {readme_file}.")

if __name__ == "__main__":
    main_optimized()
    main()

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
import chardet
import requests
from google.colab import files
from dotenv import load_dotenv  # For loading environment variables from .env file

# Step 1: Upload the .env file
print("Please upload your .env file:")
uploaded_env = files.upload()
env_file_path = list(uploaded_env.keys())[0]

if not os.path.isfile(env_file_path) or not env_file_path.lower().endswith(".env"):
    raise ValueError("A valid .env file is required.")

# Step 2: Load Environment Variables from the uploaded .env file
load_dotenv(env_file_path)  # Loads variables from the uploaded .env file

# Step 3: Read API Token from Environment Variable
api_proxy_token = os.environ.get("AIPROXY_TOKEN")
if not api_proxy_token:
    raise ValueError("API proxy token not found. Please set the 'AIPROXY_TOKEN' in the .env file.")

# Step 4: Upload the Dataset (CSV)
print("Please upload your CSV dataset file:")
uploaded_csv = files.upload()
csv_file_path = list(uploaded_csv.keys())[0]

if not os.path.isfile(csv_file_path) or not csv_file_path.lower().endswith(".csv"):
    raise ValueError("A valid CSV file is required.")

# Function to detect the encoding of a file
def detect_encoding(file_path):
    """Detect file encoding."""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(1024))  # Read the first 1 KB for detection
        return result['encoding']

# Function to read a CSV file
def read_csv(filename):
    """Read the dataset with the correct encoding."""
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
    """Analyze the dataset and return a summary."""
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
    """Generate visualizations for the dataset."""
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

# Function to interact with the LLM (via API Proxy)
def interact_with_llm_optimized(filename, analysis, api_token):
    """Interact with the gpt-4o-mini LLM via the API Proxy with reduced tokens."""
    import json  # Ensure payload and response debugging

    try:
        # API Proxy Base URL and Endpoint
        api_proxy_base_url = "https://aiproxy.sanand.workers.dev"
        api_url = f"{api_proxy_base_url}/openai/v1/chat/completions"

        # Request Headers
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        # Construct Optimized Prompt
        prompt = (
            f"I analyzed a dataset named '{filename}' with:\n"
            f"- Shape: {analysis.get('shape')}\n"
            f"- Columns: {list(analysis.get('columns').keys())}\n"
            f"- Missing Values Count: {sum(analysis.get('missing_values').values())}\n\n"
            "Summarize key insights and suggest improvements."
        )

        # Request Payload with reduced tokens
        payload = {
            "model": "gpt-4o-mini",  # The required model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,  # Reduced token limit
            "temperature": 0.5,  # Lower temperature for focused responses
            "top_p": 0.9  # Slightly narrower randomness for efficiency
        }

        # Log Payload for Debugging
        print("Payload:", json.dumps(payload, indent=4))

        # Send API Request
        response = requests.post(api_url, headers=headers, json=payload)

        # Handle Response
        if response.status_code == 200:
            result = response.json()
            # Log result for confirmation
            print("Response:", json.dumps(result, indent=4))
            return result['choices'][0]['message']['content'].strip()
        else:
            # Log detailed error
            print(f"Error Details: {response.text}")
            raise ValueError(f"API Request failed with status code {response.status_code}: {response.text}")

    except Exception as e:
        # Log traceback for debugging
        print(f"Error interacting with LLM: {e}")
        traceback.print_exc()
        return "Failed to generate insights from the LLM."

# Function to save the analysis and insights to a Markdown file
def save_markdown(analysis, charts, insights, output_file):
    """Save analysis, insights, and visualizations to a Markdown file."""
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
def main_optimized():
    """Main function to process the dataset and generate insights with minimal cost."""
    print(f"Processing {csv_file_path}...")
    df = read_csv(csv_file_path)
    if df is None:
        return

    analysis = analyze_data(df)
    output_prefix = os.path.splitext(os.path.basename(csv_file_path))[0]
    charts = visualize_data(df, output_prefix)

    # Only request insights if dataset is small/important
    if len(df) <= 10000:  # Example threshold
        insights = interact_with_llm_optimized(csv_file_path, analysis, api_proxy_token)
    else:
        insights = "Dataset too large for insights within token budget."

    readme_file = f"{output_prefix}_README.md"
    save_markdown(analysis, charts, insights, readme_file)
    print(f"Completed analysis for {csv_file_path}. Results saved to {readme_file}.")

if __name__ == "__main__":
    main_optimized()

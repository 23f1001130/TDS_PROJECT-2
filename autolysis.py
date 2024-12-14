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
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from myfile.env
load_dotenv("C:\\Users\\91735\\OneDrive\\Desktop\\Analysis\\myfile.env")

# Get the AI Proxy token from the environment variable
aiproxy_token = os.environ["AIPROXY_TOKEN"]

# Set up the OpenAI API with the token
openai.api_key = aiproxy_token

# Define file paths for CSVs and output folder
csv_folder = "C:\\Users\\91735\\OneDrive\\Desktop\\Analysis"
output_folder = os.path.join(csv_folder, "analysis_results")
os.makedirs(output_folder, exist_ok=True)

def read_csv_file(file_name):
    """Reads a CSV file and returns a DataFrame."""
    try:
        file_path = os.path.join(csv_folder, file_name)
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_name}")
        return df
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None

def analyze_data(df):
    """Analyzes the data and generates a summary."""
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }
    return summary

def generate_gpt_summary(df):
    """Generates a summary using GPT-4o-Mini."""
    data_preview = df.head(5).to_string()
    prompt = f"""
    Analyze the following dataset and provide insights:
    {data_preview}
    """
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating GPT summary: {e}")
        return None

def create_correlation_matrix(df, output_path):
    """Creates a correlation matrix heatmap and saves it as an image."""
    try:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(output_path)
        plt.close()
        print(f"Correlation matrix saved to {output_path}")
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")

def main():
    # Specify the CSV file name
    csv_file = "input_data.csv"  # Replace with your actual CSV file name

    # Read the CSV file
    df = read_csv_file(csv_file)
    if df is None:
        return

    # Analyze the data
    data_summary = analyze_data(df)
    print("Data Summary:", data_summary)

    # Generate GPT-based summary
    gpt_summary = generate_gpt_summary(df)
    if gpt_summary:
        print("GPT Summary:", gpt_summary)

    # Create and save the correlation matrix
    correlation_image_path = os.path.join(output_folder, "correlation_matrix.png")
    create_correlation_matrix(df, correlation_image_path)

    # Save the summaries to a README file
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write("# Analysis Report\n\n")
        readme_file.write("## Data Summary\n")
        readme_file.write(f"{data_summary}\n\n")
        if gpt_summary:
            readme_file.write("## GPT Insights\n")
            readme_file.write(f"{gpt_summary}\n")
    print(f"Analysis report saved to {readme_path}")

if __name__ == "__main__":
    main()

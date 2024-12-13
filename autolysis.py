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
import logging
import sys
from dotenv import load_dotenv
import requests
import plotly.express as px

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv("AIPROXY_TOKEN")

# Validate API Token
if not API_TOKEN:
    sys.exit("Environment variable AIPROXY_TOKEN is not set. Please check your .env file.")

# AI Proxy setup
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"

# Set up logging
logging.basicConfig(
    filename="autolysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function: Load dataset dynamically
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, encoding="latin1")
        logging.info("Dataset loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit("Failed to load dataset. Check the file path or format.")

# Function: Preprocess data dynamically
def preprocess_data(data):
    try:
        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=["number"])
        non_numeric_cols = data.select_dtypes(exclude=["number"])

        # Fill NaNs in numeric columns with the mean
        for col in numeric_cols.columns:
            data[col] = data[col].fillna(data[col].mean())

        # Fill NaNs in non-numeric columns with "Unknown"
        for col in non_numeric_cols.columns:
            data[col] = data[col].fillna("Unknown")

        logging.info("Data preprocessing completed.")
        return data
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        sys.exit("Data preprocessing failed.")

# Function: Analyze data dynamically
def analyze_data(data):
    try:
        numeric_cols = data.select_dtypes(include=["number"])
        categorical_cols = data.select_dtypes(exclude=["number"])

        summary = numeric_cols.describe().to_string()
        missing = data.isnull().sum()
        correlation = numeric_cols.corr()

        categorical_summary = categorical_cols.describe(include="all").to_string() if not categorical_cols.empty else "No categorical data."

        logging.info("Data analysis completed.")
        return {
            "summary": summary,
            "missing": missing,
            "correlation": correlation,
            "categorical_summary": categorical_summary,
        }
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        sys.exit("Data analysis failed.")

# Function: Visualize data dynamically (using Plotly for heatmap and top 3 histograms)
def visualize_data(data, correlation, output_dir, output_name_prefix):
    try:
        # Create and save the correlation heatmap
        fig = px.imshow(
            correlation, 
            text_auto=True, 
            title="Correlation Heatmap", 
            color_continuous_scale='Viridis'
        )
        fig.write_image(os.path.join(output_dir, f"{output_name_prefix}_correlation_heatmap.png"))
        logging.info("Correlation heatmap saved as PNG.")
        
        # Create and save top 3 histograms based on highest correlation
        numeric_cols = data.select_dtypes(include=["number"])
        top_columns = numeric_cols.corr().abs().sum().sort_values(ascending=False).head(3).index

        for col in top_columns:
            # Plot histogram using Plotly for the top 3 correlated columns
            fig = px.histogram(
                data, 
                x=col, 
                title=f"Histogram of {col}",
                nbins=30, 
                color_discrete_sequence=["#636EFA"]
            )
            fig.write_image(os.path.join(output_dir, f"{output_name_prefix}_{col}_histogram.png"))
            logging.info(f"Histogram for {col} saved as PNG.")
    
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        sys.exit("Visualization failed.")

# Function: Generate dynamic narrative using AI Proxy
def generate_narrative(data_info, include_summary=True, include_missing=True, include_correlation=True, include_categorical=True):
    try:
        # Create prompt dynamically based on the user's preference
        prompt = "You are a data analyst. Here is the dataset summary:\n\n"

        if include_summary:
            prompt += f"Dataset Summary:\n{data_info['summary']}\n\n"

        if include_missing:
            prompt += f"Missing values:\n{data_info['missing']}\n\n"

        if include_correlation:
            prompt += f"Correlation analysis (if applicable):\n{data_info['correlation']}\n\n"

        if include_categorical:
            prompt += f"Categorical data summary:\n{data_info['categorical_summary']}\n\n"

        prompt += "Write a detailed analysis story with insights, implications, and recommendations."

        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4o-mini",  # or another smaller model to optimize time
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        narrative = response.json()["choices"][0]["message"]["content"]
        logging.info("Narrative generated successfully.")
        return narrative

    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        sys.exit("Failed to generate narrative.")

# Function: Create README.md dynamically
def create_readme(narrative, output_dir, output_name_prefix):
    try:
        # Save the README in the corresponding dataset directory
        readme_path = os.path.join(output_dir, f"README.md")
        with open(readme_path, "w") as file:
            file.write("# Analysis Report\n\n")
            file.write(narrative)
            file.write("\n\n![Correlation Heatmap](goodreads_correlation_heatmap.png)\n")
            
            # Add top 3 histogram links to the README
            for col in ["col1", "col2", "col3"]:  # Replace with actual column names
                file.write(f"![{col} Histogram]({output_name_prefix}_{col}_histogram.png)\n")
            
        logging.info("README.md created successfully.")
    except Exception as e:
        logging.error(f"Error creating README.md: {e}")
        sys.exit("Failed to create README.md.")

# Main function
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: uvicorn autolysis:app --reload")

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_name_prefix = dataset_name  # Use dataset name as prefix
    
    # Create output directory based on dataset name
    output_dir = os.path.join(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset(file_path)
    processed_data = preprocess_data(data)
    analysis_results = analyze_data(processed_data)

    # Generate narrative dynamically by choosing which sections to include
    narrative = generate_narrative(
        analysis_results, 
        include_summary=True,  # User can modify these flags as needed
        include_missing=True,
        include_correlation=True,
        include_categorical=True
    )

    create_readme(narrative, output_dir, output_name_prefix)
    visualize_data(processed_data, analysis_results["correlation"], output_dir, output_name_prefix)

if __name__ == "__main__":
    main()

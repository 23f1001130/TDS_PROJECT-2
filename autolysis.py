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
#   "requests",
#   "ipykernel",
# ]
# ///
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai  # Ensure this library is installed: pip install openai

# -----------------------------------------------------------
# Function to analyze dataset properties and statistics
# -----------------------------------------------------------
def handle_data_types(df):
    """
    Automatically analyze both numeric and categorical data types.
    """
    numeric_data = df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])

    # Handle numeric data analysis
    numeric_summary = numeric_data.describe()
    correlation_matrix = numeric_data.corr()

    # Handle categorical data analysis
    categorical_summary = categorical_data.describe(include='all')

    return numeric_summary, categorical_summary, correlation_matrix


def perform_analysis(df):
    """
    Perform a comprehensive analysis of the dataset:
    - Numerical and categorical data summaries
    - Missing values check
    - Correlation matrix for numeric data
    """
    print("Performing dataset analysis...")  # Debug message
    
    # Handle both numeric and categorical data types
    numeric_summary, categorical_summary, correlation_matrix = handle_data_types(df)
    print("Generated numerical and categorical summaries.")  # Debug
    
    # Identify missing data
    missing_data = df.isnull().sum()
    print("Identified missing data counts.")  # Debug
    
    # Calculate additional statistical metrics
    mean = df.mean()
    median = df.median()
    mode = df.mode().iloc[0]
    std_dev = df.std()

    # Store the results
    stats = {
        'mean': mean,
        'median': median,
        'mode': mode,
        'std_dev': std_dev
    }
    
    print("Dataset analysis completed.")  # Debug
    return numeric_summary, categorical_summary, missing_data, correlation_matrix, stats


# -----------------------------------------------------------
# Function to detect outliers using Interquartile Range (IQR)
# -----------------------------------------------------------
def find_outliers(df):
    """
    Detect outliers in the dataset using the IQR method for numeric data.
    """
    print("Detecting outliers in numeric data...")  # Debug
    
    # Extract numeric columns
    numeric_data = df.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        print("No numeric data found for outlier detection.")  # Debug
        return pd.Series(dtype=int)

    # Calculate IQR for each column
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Identify rows that contain outliers
    outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))
    outlier_counts = outliers.sum()
    print("Outlier detection complete.")  # Debug
    return outlier_counts


# -----------------------------------------------------------
# Function to create visualizations for data insights
# -----------------------------------------------------------
def generate_visualizations(correlation_matrix, outlier_counts, df, output_directory):
    """
    Create visualizations including correlation heatmap, outlier distributions, and distribution plots.
    """
    print("Creating data visualizations...")  # Debug
    
    # Correlation heatmap
    if not correlation_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        correlation_matrix_path = os.path.join(output_directory, 'correlation_matrix.png')
        plt.savefig(correlation_matrix_path)
        plt.close()
    else:
        print("No correlation matrix generated; skipping heatmap.")  # Debug
        correlation_matrix_path = None

    # Outlier visualization
    if not outlier_counts.empty and outlier_counts.sum() > 0:
        plt.figure(figsize=(14, 7))
        outlier_counts.plot(kind='bar', color='crimson')
        plt.title('Outlier Counts by Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_path = os.path.join(output_directory, 'outliers.png')
        plt.savefig(outliers_path)
        plt.close()
    else:
        print("No significant outliers detected.")  # Debug
        outliers_path = None

    # Distribution of first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(12, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        distribution_path = os.path.join(output_directory, 'distribution.png')
        plt.savefig(distribution_path)
        plt.close()
    else:
        print("No numeric columns available for distribution plot.")  # Debug
        distribution_path = None

    print("Visualizations generated successfully.")  # Debug
    return correlation_matrix_path, outliers_path, distribution_path


# -----------------------------------------------------------
# Function to generate README file summarizing results
# -----------------------------------------------------------
def write_readme(summary, missing_data, corr_matrix, outlier_counts, df, output_directory, stats):
    """
    Create a README.md summarizing the analysis and visualizations.
    """
    print("Writing analysis results to README.md...")  # Debug
    readme_path = os.path.join(output_directory, 'README.md')
    try:
        with open(readme_path, 'w') as readme:
            readme.write("# Dataset Analysis Report\n\n")
            readme.write("## Summary of the Analysis\n")
            readme.write("This report provides an overview of the dataset, including statistical summaries, missing data, and visualizations.\n\n")

            # Adding numerical summaries
            readme.write("### Summary Statistics\n")
            readme.write(summary[0].to_string())
            readme.write("\n\n")
            
            # Adding categorical summaries
            readme.write("### Categorical Data Summary\n")
            readme.write(summary[1].to_string())
            readme.write("\n\n")
            
            # Adding statistical insights (mean, median, etc.)
            readme.write("### Additional Statistical Insights\n")
            for stat, value in stats.items():
                readme.write(f"- {stat.capitalize()}: {value}\n")
            readme.write("\n\n")
            
            # Missing data overview
            readme.write("### Missing Data\n")
            readme.write(missing_data.to_string())
            readme.write("\n\n")

            # Mention visualizations
            readme.write("### Visual Representations\n")
            readme.write("- Correlation heatmap saved as 'correlation_matrix.png'\n")
            if outlier_counts.sum() > 0:
                readme.write("- Outlier counts saved as 'outliers.png'\n")
            readme.write("- Numeric distribution plot saved as 'distribution.png'\n")

            # Additional insights
            readme.write("\n\n## Key Insights\n")
            readme.write(f"- Total number of rows: {df.shape[0]}\n")
            readme.write(f"- Columns with missing values: {missing_data[missing_data > 0]}\n")
        
        print("README file created successfully.")  # Debug
    except Exception as e:
        print(f"Error while writing README: {e}")


# -----------------------------------------------------------
# Main function to execute the full workflow
# -----------------------------------------------------------
def main(input_file):
    """
    Main function to execute data analysis and visualization workflow.
    """
    print("Starting data analysis workflow...")  # Debug
    
    # Load dataset
    try:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")  # Debug
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Perform analysis
    numeric_summary, categorical_summary, missing_data, correlation_matrix, stats = perform_analysis(df)

    # Detect outliers
    outliers = find_outliers(df)

    # Create visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(correlation_matrix, outliers, df, output_dir)

    # Write README
    write_readme([numeric_summary, categorical_summary], missing_data, correlation_matrix, outliers, df, output_dir, stats)

    print("Workflow completed successfully!")  # Debug


# -----------------------------------------------------------
# Entry point for the script
# -----------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

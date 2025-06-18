
import pandas as pd
from scipy.stats import spearmanr

# File paths
high_path = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic/well_engineered.csv"
low_path = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic/not_well_engineered.csv"
metrics_path = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/quality_metrics_and_indicators.txt"
output_path = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic/non_pythonic_well_and_not_well_eng.txt"

# Load datasets
df_high = pd.read_csv(high_path)
df_low = pd.read_csv(low_path)

# Load quality-related columns
with open(metrics_path, "r") as f:
    quality_columns = [line.strip() for line in f.readlines()]

# Ensure the metric columns exist in both datasets
quality_columns = [col for col in quality_columns if col in df_high.columns and col in df_low.columns]

# Function to compute Spearman correlation
def compute_spearman(df, label):
    results = []
    for col in quality_columns:
        series = df[[col, "pythonic_percentage"]].dropna()
        if series.empty:
            results.append(f"{label} | {col}: No data")
            continue
        rho, pval = spearmanr(series["pythonic_percentage"], series[col])
        marker = " *" if pval < 0.05 else ""
        results.append(f"{label} | {col}: rho={rho:.4f}, p={pval:.4f}{marker}")
    return results

# Compute correlations for both groups
results_high = compute_spearman(df_high, "well_eng")
results_low  = compute_spearman(df_low, "not_well_eng")

# Write results to output file
with open(output_path, "w") as f:
    for line in results_high + results_low:
        f.write(line + "\n")

print(f"Spearman analysis completed. Results saved to {output_path}")

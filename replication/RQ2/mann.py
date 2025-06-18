import pandas as pd
from scipy.stats import mannwhitneyu

df_well = pd.read_csv("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic/not_well_engineered.csv")
df_not_well = pd.read_csv("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic/well_engineered.csv")

with open("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/quality_metrics_and_indicators.txt", "r") as f:
    quality_metrics = [line.strip() for line in f if line.strip()]

results = []

for metric in quality_metrics:
    if metric in df_well.columns and metric in df_not_well.columns:
        well_vals = df_well[metric].dropna()
        not_well_vals = df_not_well[metric].dropna()
        if len(well_vals) > 0 and len(not_well_vals) > 0:
            try:
                stat, p_value = mannwhitneyu(well_vals, not_well_vals, alternative='two-sided')
                mean_well = well_vals.mean()
                mean_not_well = not_well_vals.mean()
                if p_value < 0.05:
                    direction = "Well Engineered" if mean_well > mean_not_well else "Not Well Engineered"
                    results.append([metric, mean_well, mean_not_well, p_value, direction])
            except Exception as e:
                print(f"Errore su {metric}: {e}")

df_results = pd.DataFrame(results, columns=[
    "Metric", "Mean (Well Engineered)", "Mean (Not Well Engineered)",
    "P-Value", "More frequent in "
])


df_results.to_csv("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/well_not_well/pythonic_and_not_pythonic/pythonic_and_not_pythonic_not_and_well_eng_mann.csv", index=False)
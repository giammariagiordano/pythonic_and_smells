import pandas as pd
from scipy.stats import shapiro, mannwhitneyu, ttest_ind

# Load dataset
df = pd.read_csv("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/complete.csv")

# Identify relevant smell columns
ml_smell_cols = [col for col in df.columns if col.startswith("ML_Specific_Code_Smell_")]
python_smell_cols = [col for col in df.columns if col.startswith("smell_")]

# Calculate total smells per project
df['ml_smells_total'] = df[ml_smell_cols].sum(axis=1)
df['python_smells_total'] = df[python_smell_cols].sum(axis=1)

# Normalize per 1K LOC
df['ml_smells_per_kloc'] = df['ml_smells_total'] / df['loc'] * 1000
df['python_smells_per_kloc'] = df['python_smells_total'] / df['loc'] * 1000

# Split into groups
well = df[df['Engineered.ML.Project'] == 1.0]
not_well = df[df['Engineered.ML.Project'] == 0.0]

# Function for comparison
def compare_groups(well_vals, not_well_vals, label):
    shapiro_w = shapiro(well_vals)
    shapiro_nw = shapiro(not_well_vals)

    print(f"\n--- {label} ---")
    print(f"Shapiro-Wilk test p-values:")
    print(f"  Well-engineered: {shapiro_w.pvalue:.4f}")
    print(f"  Not well-engineered: {shapiro_nw.pvalue:.4f}")

    if shapiro_w.pvalue < 0.05 or shapiro_nw.pvalue < 0.05:
        stat, pval = mannwhitneyu(well_vals, not_well_vals)
        test_used = "Mann-Whitney U test (non-parametric)"
        # Rank-biserial effect size
        n1, n2 = len(well_vals), len(not_well_vals)
        effect_size = 1 - (2 * stat) / (n1 * n2)
    else:
        stat, pval = ttest_ind(well_vals, not_well_vals, equal_var=False)
        test_used = "T-test (parametric)"
        # Cohen's d can be added here if needed
        effect_size = None

    print(f"\nStatistical test used: {test_used}")
    print(f"p-value: {pval:.6f}")
    print(f"Mean (Well-engineered): {well_vals.mean():.3f}")
    print(f"Mean (Not well-engineered): {not_well_vals.mean():.3f}")

    if pval < 0.05:
        print("=> Statistically significant difference ✅")
    else:
        print("=> No statistically significant difference ❌")

    if effect_size is not None:
        print(f"Effect size (rank-biserial |r|): {abs(effect_size):.3f}")

# Run comparisons
print("=== COMPARISON OF NORMALIZED VALUES (per 1K LOC) ===")
compare_groups(well['python_smells_per_kloc'], not_well['python_smells_per_kloc'], "Python-specific code smells per KLOC")
compare_groups(well['ml_smells_per_kloc'], not_well['ml_smells_per_kloc'], "ML-specific code smells per KLOC")
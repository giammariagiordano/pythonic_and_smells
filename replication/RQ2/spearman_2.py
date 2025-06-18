import pandas as pd
import numpy as np
from scipy.stats import spearmanr

INPUT_FILE = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/complete.csv"
OUTPUT_CORR_FILE = "correlation_control_variables.csv"
OUTPUT_LATEX_FILE = "correlation_control_variables.tex"
SPARSITY_THRESHOLD = 0.95

df = pd.read_csv(INPUT_FILE)

ml_smells = [c for c in df.columns if c.startswith("ML_Specific_Code_Smell_")]
traditional_smells = [c for c in df.columns if c.startswith("smell_")]
all_smells = ml_smells + traditional_smells

control_vars = [
    "loc",
    "Commits",
    "file_complexity"
]

df_filtered = df[df["loc"] > 0].copy()

zero_frac_smells = (df_filtered[all_smells] == 0).mean()
smells_kept = zero_frac_smells[zero_frac_smells < SPARSITY_THRESHOLD].index.tolist()

results = []
m_tests = len(control_vars) * len(smells_kept)
alpha1 = 0.05 / m_tests
alpha2 = 0.01 / m_tests
alpha3 = 0.001 / m_tests

for ctrl in control_vars:
    series_ctrl = df_filtered[ctrl].dropna()
    for smell in smells_kept:
        both = pd.concat([series_ctrl, df_filtered[smell]], axis=1).dropna()
        rho, pval = spearmanr(both[ctrl], both[smell])
        if pval < alpha3:
            stars = '***'
        elif pval < alpha2:
            stars = '**'
        elif pval < alpha1:
            stars = '*'
        else:
            stars = ''
        results.append({
            "Control": ctrl,
            "Smell": smell,
            "Spearman_rho": round(rho, 2),
            "p_value": f"{pval:.2e}",
            "Significance": stars
        })

results_df = pd.DataFrame(results)
results_df["abs_rho"] = results_df["Spearman_rho"].abs()
results_df = results_df.sort_values(by="abs_rho", ascending=False).drop(columns="abs_rho")

# === 5. SALVA CSV ===
results_df.to_csv(OUTPUT_CORR_FILE, index=False)
print(f"Correlazioni control vars vs smells salvate in: {OUTPUT_CORR_FILE}")

latex_table = results_df.to_latex(
    index=False,
    columns=["Control", "Smell", "Spearman_rho", "p_value", "Significance"],
    header=["Control Variable", "Smell", "Spearman $\\rho$", "$p$-value", "Significance (Bonferroni)"],
    longtable=False,
    escape=False
)

with open(OUTPUT_LATEX_FILE, "w") as f:
    f.write(latex_table)

print(f"Tabella LaTeX salvata in: {OUTPUT_LATEX_FILE}")
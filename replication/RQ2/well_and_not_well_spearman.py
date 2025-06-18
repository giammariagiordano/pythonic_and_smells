import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# === CONFIGURAZIONE ===
INPUT_FILE = "/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/complete.csv"
CONTROL_VARS = ["loc", "Commits", "file_complexity"]
SPARSITY_THRESHOLD = 0.95

# Funzione per trasformare nomi in etichette leggibili
def make_readable(name: str) -> str:
    return (name
            .replace("PY_", "")
            .replace("refactorizable_", "")
            .replace(".", " ")
            .replace("_", " ")
            .title())

# Carica il dataset completo
df_full = pd.read_csv(INPUT_FILE)

# Identifica i code smell (ML-specifici e tradizionali)
ml_smells = [c for c in df_full.columns if c.startswith("ML_Specific_Code_Smell_")]
traditional_smells = [c for c in df_full.columns if c.startswith("smell_")]
ALL_SMELLS = ml_smells + traditional_smells

# Identifica le feature Pythonic e Refactorizable
pythonic_features = [c for c in df_full.columns if c.startswith("PY_")]
refactorizable_features = [c for c in df_full.columns if c.startswith("refactorizable_")]
ALL_CONSTRUCTS = pythonic_features + refactorizable_features

# Crea dizionario etichette leggibili per constructs e controlli
readable_constructs = {
    f"{c}_norm": f"{make_readable(c)} Norm"
    for c in ALL_CONSTRUCTS
}
readable_controls = {cv: make_readable(cv) for cv in CONTROL_VARS}
VARIABLE_LABELS = {**readable_constructs, **readable_controls}

# Elenco completo di variabili da correlare
ALL_VARS = list(readable_constructs.keys()) + CONTROL_VARS

# Funzione che calcola correlazioni e salva CSV per un sottoinsieme
def compute_and_save(df: pd.DataFrame, subset_label: str):
    # 1. Filtra loc > 0 e normalizza i constructs
    df_sub = df[df["loc"] > 0].copy()
    for col in ALL_CONSTRUCTS:
        df_sub[f"{col}_norm"] = df_sub[col] / df_sub["loc"]

    # 2. Rimuove smells troppo sparsi (>95% zeri)
    zero_frac = (df_sub[ALL_SMELLS] == 0).mean()
    smells_kept = zero_frac[zero_frac < SPARSITY_THRESHOLD].index.tolist()

    # 3. Calcola soglie Bonferroni
    m_tests = len(ALL_VARS) * len(smells_kept)
    alpha1 = 0.05 / m_tests
    alpha2 = 0.01 / m_tests
    alpha3 = 0.001 / m_tests

    # 4. Calcola correlazioni di Spearman
    results = []
    for var in ALL_VARS:
        series_var = df_sub[var].dropna()
        for smell in smells_kept:
            both = pd.concat([series_var, df_sub[smell]], axis=1).dropna()
            rho, pval = spearmanr(both[var], both[smell])
            if pval < alpha3:
                stars = "***"
            elif pval < alpha2:
                stars = "**"
            elif pval < alpha1:
                stars = "*"
            else:
                stars = ""
            results.append({
                "Variable": VARIABLE_LABELS[var],
                "Smell": (smell
                          .replace("ML_Specific_Code_Smell_", "")
                          .replace("smell_", "")
                          .replace("_", " ")
                          .title()),
                "Spearman_rho": round(rho, 2),
                "p_value": f"{pval:.2e}",
                "Significance": stars
            })

    # 5. Ordina per valore assoluto di rho e salva CSV
    df_results = pd.DataFrame(results)
    df_results["abs_rho"] = df_results["Spearman_rho"].abs()
    df_results = df_results.sort_values("abs_rho", ascending=False).drop(columns="abs_rho")

    csv_path = f"combined_correlations_{subset_label}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


# 6. Esegui per Engineered.ML.Project = 1 e = 0
for value in (1, 0):
    df_subset = df_full[df_full["Engineered.ML.Project"] == value]
    label = f"engineered{value}"
    compute_and_save(df_subset, label)
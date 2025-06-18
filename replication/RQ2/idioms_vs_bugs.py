import pandas as pd
from scipy.stats import spearmanr


df = pd.read_csv("/Users/broke31/Desktop/jss_replication/splitted_pythonic_and_not_pythonic/complete.csv")


df["Engineering_Level"] = df["Engineered.ML.Project"].apply(
    lambda x: "Well Engineered" if x == 1 else "Not Well Engineered"
)

corr_vars = [
    "refactorizable_Assign.Multi.Targets", "refactorizable_Call.Star",
    "refactorizable_List.Comprehension", "refactorizable_Dict.Comprehension",
    "refactorizable_Set.Comprehension", "refactorizable_Truth.Value.Test",
    "refactorizable_Chain.Compare", "refactorizable_For.Multi.Targets",
    "refactorizable_For.Else", "PY_Assign.Multi.Targets", "PY_Call.Star",
    "PY_List.Comprehension", "PY_Dict.Comprehension", "PY_Set.Comprehension",
    "PY_Truth.Value.Test", "PY_Chain.Compare", "PY_For.Multi.Targets",
    "PY_For.Else", "Commits", "complexity", "ncloc", "bugs"
]

def significance_stars(pval):
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.1:
        return "*"
    else:
        return ""

def compute_annotated_spearman(data, target, variables):
    results = []
    for var in variables:
        if var != target:
            rho, pval = spearmanr(data[target], data[var])
            if pval < 0.1:  # filtro minimo
                stars = significance_stars(pval)
                results.append((var, rho, pval, stars))
    return pd.DataFrame(results, columns=["Variable", "Spearman_rho", "p_value", "Significance"])

well_df = df[df["Engineered.ML.Project"] == 1]
not_well_df = df[df["Engineered.ML.Project"] == 0]

well_corr = compute_annotated_spearman(well_df, "bugs", corr_vars)
not_well_corr = compute_annotated_spearman(not_well_df, "bugs", corr_vars)

well_corr.to_csv("spearman_well_engineered_significant.csv", index=False)
not_well_corr.to_csv("spearman_not_well_engineered_significant.csv", index=False)

print("CSV esportati con livelli di significatività annotati (*, **, ***)")
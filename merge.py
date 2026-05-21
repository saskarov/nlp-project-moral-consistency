import pandas as pd
import json

# Load the two files
df_emb  = pd.read_csv("consistency_scores_embeddings.csv")
df_cons = pd.read_csv("data/consistency_scores.csv")

# Merge embedding scores into the main consistency dataframe
df_cons = df_cons.drop(columns=["embedding_consistency"], errors="ignore")
df_merged = pd.merge(
    df_cons,
    df_emb[["scenario_id", "model_alias", "embedding_consistency"]],
    on=["scenario_id", "model_alias"],
    how="left"
)

# Save updated file
df_merged.to_csv("data/consistency_scores.csv", index=False)
print("Merged. Embedding consistency summary:")
print(df_merged.groupby(["category", "model_alias"])["embedding_consistency"].mean().round(3))
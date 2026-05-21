import json
import pandas as pd

# Load embedding scores
df_emb = pd.read_csv("consistency_scores_embeddings.csv")

# Load existing summary
with open("data/analysis_summary.json") as f:
    summary = json.load(f)

# Compute mean embedding consistency per model
for model in ["3B", "8B"]:
    sub = df_emb[df_emb["model_alias"] == model]
    mean_emb = float(sub["embedding_consistency"].mean())
    summary[model]["embedding_consistency_mean"] = round(mean_emb, 4)
    print(f"  {model}: embedding_consistency_mean = {mean_emb:.4f}")

# Compute per category
for cat in ["commonsense", "justice", "deontology", "virtue", "utilitarianism"]:
    for model in ["3B", "8B"]:
        sub = df_emb[(df_emb["category"] == cat) & (df_emb["model_alias"] == model)]
        if len(sub) > 0:
            summary["by_category"][cat][f"embedding_consistency_{model}"] = round(float(sub["embedding_consistency"].mean()), 4)

# Save updated summary
with open("data/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nanalysis_summary.json updated.")
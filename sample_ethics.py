"""
sample_ethics.py
================
Downloads (or loads from disk) the ETHICS dataset and produces a
balanced sample of scenarios for Project P10.

USAGE
-----
# If you have already downloaded ethics.tar and extracted it:
python sample_ethics.py --data_dir /path/to/ethics

# To see all options:
python sample_ethics.py --help

OUTPUT
------
data/ethics_sample.csv   — the sampled scenarios, ready for paraphrasing
data/ethics_sample.json  — same data, JSON format for easier inspection

Each row contains:
  id          — unique identifier   e.g. commonsense_0042
  category    — one of: commonsense | justice | deontology | virtue | utilitarianism
  label       — original binary label (1 = morally acceptable, 0 = not)
  scenario    — the raw scenario text from the dataset
"""

import os
import argparse
import json
import random
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────

CATEGORIES = {
    "commonsense":    {"file": "cm_train.csv",          "text_col": 1, "label_col": 0},
    "justice":        {"file": "justice_train.csv",     "text_col": 1, "label_col": 0},
    "deontology":     {"file": "deontology_train.csv",  "text_col": 1, "label_col": 0},
    "virtue":         {"file": "virtue_train.csv",      "text_col": 1, "label_col": 0},
    "utilitarianism": {"file": "util_train.csv",        "text_col": 0, "label_col": None},
}

SAMPLES_PER_CATEGORY = 20   # 20 × 5 categories = 100 scenarios total
RANDOM_SEED = 42


# ── Loaders (one per format) ─────────────────────────────────────────────────

def load_commonsense(path):
    df = pd.read_csv(path)
    # columns: label, text (short scenarios, binary)
    return [
        {"scenario": str(row.iloc[1]).strip(), "label": int(row.iloc[0])}
        for _, row in df.iterrows()
        if str(row.iloc[1]).strip()
    ]

def load_justice(path):
    df = pd.read_csv(path)
    return [
        {"scenario": str(row.iloc[1]).strip(), "label": int(row.iloc[0])}
        for _, row in df.iterrows()
        if str(row.iloc[1]).strip()
    ]

def load_deontology(path):
    df = pd.read_csv(path)
    # columns: label, scenario, excuse — we join them for context
    rows = []
    for _, row in df.iterrows():
        scenario = str(row.iloc[1]).strip()
        excuse   = str(row.iloc[2]).strip()
        combined = f"{scenario} | Excuse: {excuse}"
        rows.append({"scenario": combined, "label": int(row.iloc[0])})
    return rows

def load_virtue(path):
    df = pd.read_csv(path)
    # columns: label, scenario (trait judgments)
    return [
        {"scenario": str(row.iloc[1]).strip(), "label": int(row.iloc[0])}
        for _, row in df.iterrows()
        if str(row.iloc[1]).strip()
    ]

def load_utilitarianism(path):
    df = pd.read_csv(path, header=None)
    # columns: more_pleasant, less_pleasant (pairwise comparisons)
    rows = []
    for _, row in df.iterrows():
        more = str(row.iloc[0]).strip()
        less = str(row.iloc[1]).strip()
        # Frame as a comparison scenario; label=1 means first is better
        combined = f"More pleasant: {more} | Less pleasant: {less}"
        rows.append({"scenario": combined, "label": 1})
    return rows


LOADERS = {
    "commonsense":    load_commonsense,
    "justice":        load_justice,
    "deontology":     load_deontology,
    "virtue":         load_virtue,
    "utilitarianism": load_utilitarianism,
}


# ── Main sampling logic ───────────────────────────────────────────────────────

def sample_category(data_dir, category, n, seed):
    cfg    = CATEGORIES[category]
    path   = os.path.join(data_dir, category, cfg["file"])

    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path}")
        return []

    loader = LOADERS[category]
    rows   = loader(path)
    print(f"  {category}: {len(rows)} rows loaded")

    # Balanced sample: equal positive/negative where possible
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]

    random.seed(seed)
    half = n // 2
    sampled = (
        random.sample(pos, min(half, len(pos))) +
        random.sample(neg, min(n - half, len(neg)))
    )
    random.shuffle(sampled)

    # Attach metadata
    for i, item in enumerate(sampled):
        item["id"]       = f"{category}_{i:04d}"
        item["category"] = category
        item["paraphrases"] = []  # placeholder for Phase 2

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample ETHICS dataset for P10")
    parser.add_argument(
        "--data_dir", type=str, default="./ethics",
        help="Root directory of the extracted ethics dataset"
    )
    parser.add_argument(
        "--n_per_category", type=int, default=SAMPLES_PER_CATEGORY,
        help="Number of scenarios to sample per category (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./data",
        help="Output directory for sampled files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_samples = []
    print(f"\nSampling {args.n_per_category} scenarios from each of {len(CATEGORIES)} categories...\n")

    for category in CATEGORIES:
        samples = sample_category(args.data_dir, category, args.n_per_category, args.seed)
        all_samples.extend(samples)

    total = len(all_samples)
    print(f"\nTotal scenarios sampled: {total}")

    # ── Save as CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "ethics_sample.csv")
    df_out = pd.DataFrame([
        {"id": s["id"], "category": s["category"], "label": s["label"], "scenario": s["scenario"]}
        for s in all_samples
    ])
    df_out.to_csv(csv_path, index=False)
    print(f"CSV saved  →  {csv_path}")

    # ── Save as JSON (includes paraphrases placeholder) ──────────────────────
    json_path = os.path.join(args.out_dir, "ethics_sample.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"JSON saved →  {json_path}")

    # ── Quick summary ────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    for cat in CATEGORIES:
        cat_rows = df_out[df_out["category"] == cat]
        pos = (cat_rows["label"] == 1).sum()
        neg = (cat_rows["label"] == 0).sum()
        print(f"  {cat:<18} {len(cat_rows):>3} scenarios  (pos={pos}, neg={neg})")
    print(f"  {'TOTAL':<18} {total:>3}")
    print("─────────────────────────────────────────────────────────────\n")
    print("Next step: run  generate_paraphrases.py  on ethics_sample.json")


if __name__ == "__main__":
    main()

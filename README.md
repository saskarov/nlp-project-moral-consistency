# Right, Wrong, and Everything in Between

**Evaluating Moral Consistency and Linguistic Strategy in Large Language Models**

NLP final project (P10) — Università degli Studi di Milano, Data Science for Economics.

This project investigates whether Large Language Models give *consistent* moral judgements when the same ethical scenario is reworded, and what linguistic strategies (hedging, modal verbs, refusals) they use when reasoning about morality. Two instruction-tuned Llama models of different sizes (Llama 3.2-3B and Llama 3.1-8B) are compared across five categories of the ETHICS benchmark.

## Key Findings

- The smaller **3B model is more morally consistent** than the larger 8B model on four of five ethical categories — consistency does not scale with model size.
- **Consistency is domain-dependent**: deontology shows the largest gap between models (3B: 0.83, 8B: 0.50 verdict consistency), while utilitarianism is hardest for both.
- Roughly **23% of paraphrase prompts were refused** by the paraphrase-generating model itself — a methodological finding about LLM-based pipeline contamination.
- **Hedging does not predict consistency**: linguistic uncertainty and semantic consistency are orthogonal dimensions.

## Pipeline Overview

```
sample_ethics.py          ->  data/ethics_sample.json        (Phase 1: sample ETHICS)
generate_paraphrases_1.py ->  data/ethics_paraphrased.json   (Phase 2: paraphrase via Ollama)
query_models_1.py         ->  data/ethics_responses.json     (Phase 3: query target models)
analyze.py                ->  data/*.csv, plots/*.png         (Phase 4: analysis & figures)
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── sample_ethics.py            # Phase 1 — sample & balance ETHICS scenarios
├── generate_paraphrases_1.py   # Phase 2 — generate paraphrases via Ollama
├── query_models_1.py           # Phase 3 — collect moral judgements from both models
├── analyze.py                  # Phase 4 — consistency metrics, linguistic markers, plots
├── data/
│   ├── ethics_sample.json      # 90 sampled scenarios
│   ├── ethics_paraphrased.json # scenarios + 4 paraphrases each
│   ├── ethics_responses.json   # all model responses (468 records)
│   ├── consistency_scores.csv  # per-scenario consistency metrics
│   ├── linguistic_markers.csv  # per-response linguistic annotations
│   └── analysis_summary.json   # aggregated results
└── plots/                      # generated figures
```

## Setup

This project requires **Python 3.11 or 3.12** (PyTorch does not yet support 3.13) and a local [Ollama](https://ollama.com) installation for the generation steps.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull the two models
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
ollama pull hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M

# 3. Download the ETHICS dataset
wget https://people.eecs.berkeley.edu/~hendrycks/ethics.tar
tar -xf ethics.tar
```

## Reproducing the Experiments

```bash
# Phase 1 — sample 20 scenarios per category (100 total, seed 42)
python sample_ethics.py --data_dir ./ethics --n_per_category 20 --out_dir ./data

# Phase 2 — generate 4 paraphrases per scenario (uses the 3B model)
python generate_paraphrases_1.py

# Phase 3 — query both models for moral judgements (~1000 calls)
python query_models_1.py \
  --models "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M" \
           "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"

# Phase 4 — run full analysis and generate all plots
python analyze.py --input data/ethics_responses.json
```

To regenerate only the plots from existing scores (no model calls needed):

```bash
python analyze.py --plots_only
```

## Metrics

**Embedding consistency** — mean pairwise cosine similarity of SBERT (`all-MiniLM-L6-v2`) embeddings of the responses to a scenario's paraphrases.

**Verdict consistency** — fraction of responses sharing the majority verdict (acceptable / unacceptable / ambiguous / refusal), where verdicts are extracted via a rule-based classifier.

## Data Sources

- **ETHICS benchmark** — Hendrycks et al. (2021), [github.com/hendrycks/ethics](https://github.com/hendrycks/ethics)

## References

1. Bonagiri et al. (2024). *SaGE: Evaluating Moral Consistency in Large Language Models.* [arXiv:2402.13709](https://arxiv.org/abs/2402.13709)
2. Forbes et al. (2020). *Social Chemistry 101: Learning to Reason about Social and Moral Norms.* EMNLP.
3. Hendrycks et al. (2021). *Aligning AI with Shared Human Values.* ICLR. [arXiv:2008.02275](https://arxiv.org/abs/2008.02275)
4. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.

## AI Usage Disclaimer

Parts of this project were developed with the assistance of Claude (Anthropic), used to support development of the experimental pipeline, structuring of methodological workflows, drafting of descriptive texts, and identification of relevant datasets. All AI-assisted content has been reviewed, edited, and validated by the author, who takes full responsibility for the final content.

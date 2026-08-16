# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Yash Raj Bhatnagar
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** flyrank-task-1
- **Date:** August 2026

## 1. Problem framing
This project supports the decision of prioritizing historical content for editorial refreshes. The unit of analysis is the individual URL/page. The output is a ranked action playbook (score and categorical action: Full Refresh, Meta Optimization, or Monitor) that an editorial team can use to assign work. The cost of a wrong call is wasted editorial hours on pages that cannot recover, or destabilizing a healthy evergreen page. Machine learning helps here because static heuristics (like CTR gap formulas) fail to capture the non-linear realities of how age, ranking depth, and traffic scale interact.

## 2. Data safety
The dataset used is `fact_content_daily_performance` from the FlyRank warehouse. 
* **Excluded Columns:** All identifier columns (`client_hash_id`, `content_hash_id`) were excluded from the feature matrix to ensure the model learns environmental patterns, not pseudonymous URL memorization. 
* **Leakage Risks:** To prevent target leakage, all CTR-derived fields (`feat_ctr`, `expected_ctr`, `ctr_gap`) were deliberately removed from the model's features, as they were used to define the target label. 
* I confirm no client-identifying details or PII exist anywhere in `work/`.

## 3. Baseline
The baseline is the "Expected Clicks Lost" Opportunity Score heuristic, calculated as: `Impressions * CTR Gap * log2(1 + Days Stale)`. It is a fair comparison because it relies on the exact same base data (traffic, staleness, position benchmarks) as the ML model. Evaluated on the validation split, this baseline achieved a PR-AUC of 0.224.

## 4. Model / analysis
I used a Random Forest Classifier because tree-based ensembles excel at capturing complex, non-linear interactions (e.g., how high staleness affects a Position 2 page very differently than a Position 8 page) without requiring monotonic feature scaling. 
* **Features Included:** `gsc_avg_position`, `gsc_impressions`, `feat_days_stale`.
* **Features Excluded:** Direct CTR metrics were left out on purpose to prevent leakage.
* **Target Definition:** `target_degraded` is a binary flag indicating if a page falls into the bottom 25th percentile for CTR relative to its specific ranking position cohort, provided it underperforms the expected benchmark.

## 5. Evaluation
* **Split Design:** A `GroupShuffleSplit` (grouped by `content_id`) with an 80/20 ratio. Grouping by ID is critical to prevent temporal data from the same URL leaking across the train and validation sets.
* **Base Rate:** The majority class is healthy content; the target base rate (degraded pages) is highly imbalanced at ~7.8%.
* **Metrics vs Baseline:** Because of the class imbalance, Precision-Recall AUC (PR-AUC) was used instead of ROC-AUC. 
  * Baseline Heuristic PR-AUC: 0.224
  * Random Forest PR-AUC: 0.777
* **Error Analysis:** The most common False Positives occur on deep long-tail pages (e.g., Positions 75+ with ~50 impressions). The model flags them as degraded, but near-zero CTR is mathematically normal at that ranking depth.

## 6. Interpretation
The Random Forest found that traffic scale (`gsc_impressions`) is the overwhelmingly dominant signal for predicting degradation probability (Gini Importance > 0.8), followed by `gsc_avg_position`. Interestingly, `feat_days_stale` had the lowest standalone importance, meaning age alone does not predict degradation unless combined with massive traffic footprints and declining positions. The "no effect" surprise here was that staleness isn't as universally detrimental as traditional SEO advice claims.

## 7. Recommendation
The model outputs a ranked editorial playbook categorized into:
1. **Full Content Refresh:** (>50% degradation probability + >90 days stale). 
2. **Meta Optimization Only:** (>50% probability + <90 days stale).
3. **Monitor / Maintain:** (<50% probability).
A FlyRank editor would pull the top 20 URLs from the "Full Refresh" queue and assign them to writers tomorrow. 
**Confidence & Limits:** This is a directional, decision-support tool. It predicts the *probability of degradation*, but makes no causal claims that rewriting the content will guarantee traffic recovery.

## 8. Reproducibility
* **Environment:** Executed via Google Colab with `duckdb`, `pandas`, `scikit-learn`, and `matplotlib`. 
* **Random Seeds:** `random_state=42` applied to both `GroupShuffleSplit` and `RandomForestClassifier`.
* **Execution:** A reviewer with a fresh clone can reproduce all charts and playbook tables by running `work/notebooks/capstone.ipynb` top-to-bottom.

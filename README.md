# AI auditors

Code and figures for **Can AI Help Us Oversee AI?**, the [AI Auditors Google Doc](https://docs.google.com/document/d/1HPnvlCS1kT8hW4pWavpFMFiOkkkuq8lNs3NGq8-RnMM/edit).

This repository is scoped to the draft's main analysis as read on October 9, 2026: three MALT figures and two R-Judge figures. It includes the five displayed model variants, aggregate results, plotting code, prompt templates, and methods. Collection infrastructure, failed API attempts, historical pilots, unused plots, draft comments, and raw agent transcripts remain in the private research workspace.

## Rebuild the figures

Tested with Python 3.9.6 and the package versions in `requirements.txt`. From a clean clone:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/master.py
```

The build works offline once dependencies are installed. It makes no model calls and needs no credentials or record-level data. It recomputes operating points from the frozen aggregate ROC vertices, recalculates API costs from aggregate token usage and pinned prices, and renders the figures. This is reproduction from saved results, not recollection or reconstruction of the original transcripts.

## Figures in the post

| Draft discussion | Output (PNG and PDF) | Script |
|---|---|---|
| Detecting reward hacking and sandbagging at 5% FPR | [malt_auditor_panels_tpr](output/malt_auditor_panels_tpr.png) | [make_malt_auditor_panels_tpr.py](code/make_malt_auditor_panels_tpr.py) |
| False alarms at 80% detection, by MALT behavior | [malt_auditor_panels_fpr](output/malt_auditor_panels_fpr.png) | [make_malt_auditor_panels_fpr.py](code/make_malt_auditor_panels_fpr.py) |
| MALT AUROC versus API cost, with price frontier | [malt_auditor_cost_auroc](output/malt_auditor_cost_auroc.png) | [make_malt_auditor_cost_auroc.py](code/make_malt_auditor_cost_auroc.py) |
| Detecting unsafe R-Judge interactions at 5% FPR | [rjudge_auditor_tpr5](output/rjudge_auditor_tpr5.png) | [make_rjudge_auditor_tpr5.py](code/make_rjudge_auditor_tpr5.py) |
| R-Judge false alarms at 95% and 99% detection | [rjudge_auditor_panels_fpr](output/rjudge_auditor_panels_fpr.png) | [make_rjudge_auditor_panels_fpr.py](code/make_rjudge_auditor_panels_fpr.py) |

The plotted variants are GPT-5 Nano (medium), GPT-5 (medium), GPT-6 Astra (high), Gemini 3.8 Flash (high), and Opus 4.8 (high). Earlier minimal/low variants are omitted from these figures.

`output/figure_data/` contains only aggregate numeric exports: ROC vertices, plotted values, sample counts, token totals, frozen prices/provenance, and supporting summary results. It contains no individual logs or generated explanations. `code/run_auditor_metrics.py` produces a compact cross-dataset metrics table and the post's illustrative manual-review arithmetic.

## Data and methods

Read [methods and limitations](notes/methods.md) and the [completed draft-claim audit](notes/draft_checks.md). The Google Doc is the live manuscript; this repository does not mirror its comments or archived prose.

The audit independently verifies the reported metrics and documents corrections to the ensemble, prevalence, replication, and refusal claims. [Operating-point checks](output/figure_data/draft_operating_points.csv) compare all 35 displayed fixed-rate metrics to single-cutoff alternatives. `code/check_draft_operating_points.py` rebuilds these checks from the bundled aggregate ROC data as part of the master script.

- **MALT:** [METR report](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/) and [gated public dataset](https://huggingface.co/datasets/metr-evals/malt-public). Our comparison uses manually reviewed public logs. Aggregate each model's chunk scores by maximum, requiring all chunks to be valid.
- **R-Judge:** [paper](https://aclanthology.org/2024.findings-emnlp.79/) and [dataset repository](https://github.com/Lordog/R-Judge). Our pinned repository snapshot contains 571 records; the paper describes 569. Each complete interaction gets one score.

Scores are on a 0–10 scale. Fixed-rate metrics interpolate empirical ROC curves; a single deterministic cutoff may not achieve the displayed rate. The operating points describe the evaluated samples, not thresholds calibrated on held-out deployment data. The MALT cost figure uses **October 2, 2026** prices, not live prices.

## Optional record-level analysis

Record-level analysis files remain outside Git. To recompute individual and ensemble metrics, obtain the saved research CSVs from the project owner and place them in this clone's `modified_data/`:

- `malt_chunk_analysis.csv` → `python code/run_malt_ensemble_analysis.py`
- `rjudge_analysis.csv` → `python code/run_rjudge_analysis.py`

These are the completed scored analysis files, not the upstream datasets. Downloading the original benchmark alone does not recreate our model responses. Both optional scripts are offline, preserve reasoning variants separately, and keep detailed record-level outputs ignored. They do not change the frozen figure inputs automatically.

The completed MALT ensemble experiments are mean raw score and mean percentile across Astra, Gemini, and Opus, after taking each model's maximum across chunks. Majority voting and AI-judge aggregation are not established by the saved completed results included here.

After the optional MALT analysis, `python code/check_malt_ensemble_bootstrap.py` recomputes the [paired uncertainty check](output/figure_data/malt_ensemble_bootstrap.csv) from local run scores. Its 2,000 seeded, label-stratified log resamples condition on the completed predictions; they do not account for task clustering, model selection, or deployment shift. This optional check makes no API calls.

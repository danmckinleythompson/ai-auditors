# Draft-claim validation

Audit completed October 9, 2026, against the [current Google Doc](https://docs.google.com/document/d/1HPnvlCS1kT8hW4pWavpFMFiOkkkuq8lNs3NGq8-RnMM/edit), saved predictions and requests, original dataset metadata, and primary reports. No paid calls were made. The manuscript and five figures were not edited.

The headline numbers reproduce. Several interpretations need qualification, and the ensemble and refusal descriptions need factual corrections.

## 1. Ensemble experiments: correct the methods and narrow the conclusion

Only two completed methods were found: mean raw score and mean percentile, combining Astra, Gemini, and Opus after taking each model's maximum over chunks. The local judge files contain preparation and cost estimates, explicitly marked as having no paid judge calls. No completed majority-vote implementation or judge predictions were found. R-Judge is a separate benchmark, not an AI-judge ensemble experiment.

Independent recomputation from the canonical chunk data reproduced all 32 saved ensemble/individual metric rows. Comparisons below use the same complete logs: 741 reward-hacking records and 1,755 sandbagging records.

| Behavior | Approach | AUROC | TPR at 5% FPR |
|---|---|---:|---:|
| Reward hacking | GPT-5 medium, strongest individual | 0.9762 | 97.87% |
| Reward hacking | Astra, strongest ensemble constituent | 0.9346 | 89.53% |
| Reward hacking | Mean raw score | 0.9615 | 86.57% |
| Reward hacking | Mean percentile | 0.9592 | 73.60% |
| Sandbagging | Astra, strongest individual | 0.9274 | 81.16% |
| Sandbagging | Mean raw score | 0.9320 | 81.57% |
| Sandbagging | Mean percentile | 0.9322 | 81.84% |

The ensembles improve sandbagging detection by 0.41 and 0.68 percentage points. A new paired, label-stratified log bootstrap (2,000 replicates; seed 20261009) gives 95% intervals of [-0.28, 1.07] and [-0.19, 1.50] percentage points. Sandbagging AUROC gains are modest but their corresponding intervals exclude zero: +0.0046 [0.0015, 0.0079] and +0.0048 [0.0014, 0.0084]. These exploratory intervals condition on the predictions and chosen baselines; they omit task clustering, model selection, and deployment shift. Do not interpret them as a definitive generalization test.

**Suggested replacement:** “We combined Astra, Gemini, and Opus by averaging their scores or their percentile ranks. At a 5% false-positive rate, neither approach matched the best individual model on reward hacking. On sandbagging, detection improved by less than one percentage point, although both ensembles modestly improved AUROC.”

Evidence: [saved matched metrics](../output/figure_data/malt_ensemble_reference_metrics.csv), [ensemble code](../code/run_malt_ensemble_analysis.py), [bootstrap results](../output/figure_data/malt_ensemble_bootstrap.csv), [bootstrap assumptions](../output/figure_data/malt_ensemble_bootstrap.json). Original workspace evidence: `notes/malt_judge_costs.md`, `code/master.py` judge-preparation block, and `code/check_malt_ensemble_analysis.py`.

## 2. Reported operating points: verified, but interpolation matters

All 35 fixed-rate values in the bundled cross-dataset summary reproduce from ROC vertices. An independent calculation directly from R-Judge record scores also matches all 15 displayed R-Judge rates to 1e-12. The statements about Opus detecting nearly 96%, Gemini's FPR below 4% at 95% recall, and Astra's roughly 15% FPR at 99% recall are correct **under the plotted interpolation convention**.

Scores have ties. A point between cutoffs corresponds to randomly including part of a tied-score group, achieving the displayed rates in expectation. It need not be attainable using one score cutoff. The difference changes some comparisons:

| R-Judge model | Interpolated TPR at 5% FPR | Best single-cutoff TPR with FPR ≤5% | Interpolated FPR at 99% TPR | Single-cutoff FPR with TPR ≥99% |
|---|---:|---:|---:|---:|
| Nano medium | 58.34% | 48.50% | 81.05% | 100.00% |
| GPT-5 medium | 93.02% | 89.70% | 25.86% | 32.22% |
| Astra high | 90.64% | 87.04% | 15.12% | 15.93% |
| Gemini high | 95.07% | 95.00% | 80.74% | 100.00% |
| Opus high | 95.83% | 94.33% | 60.34% | 72.59% |

Thus four models exceed 90% detection under interpolation; two do under a single cutoff constrained to ≤5% FPR. Gemini has the highest single-cutoff detection in that comparison. At ≥95% recall, single-cutoff FPRs are Gemini 3.70%, Opus 5.19%, Astra 9.26%, GPT-5 10.37%, and Nano 59.63%; the “nearly all below 10%” wording also depends on interpolation.

For MALT at 80% recall, the best interpolated FPRs are GPT-5's 2.88% for reward hacking and Astra's 2.56% for sandbagging. Requiring a single cutoff gives best FPRs of 3.46% and 2.74%, respectively, both from Astra. GPT-5's single-cutoff reward-hacking FPR is 4.90%. The existing plots are mathematically consistent; a practical deployment claim should distinguish these policies.

**Suggested methods sentence:** “We interpolate between score thresholds to compare models at a common false-positive or detection rate. These operating points can require random selection among tied scores and may not be achievable with a single cutoff.”

With approximately 300 unsafe R-Judge records, one case changes recall by about 0.33 percentage points. The thresholds are selected and assessed on the same sample; 99% recall here is not a calibrated guarantee on new data. The [paper](https://aclanthology.org/2024.findings-emnlp.79.pdf), §§2–3 and Appendix B.2, labels complete interactions and deliberately curates safe/unsafe examples. Say “unsafe interactions” rather than treating these as measured prevention rates for individual actions.

Evidence: [all operating points, cutoffs, and confusion counts](../output/figure_data/draft_operating_points.csv), [offline verification script](../code/check_draft_operating_points.py). The independent record-level audit retained all 270 safe records for every model; Gemini and Opus each lack one unsafe response. Their denominators are 300 unsafe records, versus 301 for the other models.

## 3. Human review and historical replication: qualify

All 7,179 metadata rows were reread directly from the 42 downloaded public MALT Parquet shards. There are 1,826 reviewed logs and 5,353 unreviewed logs; our analyses use only reviewed logs. The behavior comparisons include 1,811 distinct logs after excluding 15 hardcoded-only cases. The draft's statement that humans reviewed all logs needs “in our analysis” or “the reviewed subset.” METR's published experiment also includes nonpublic data. [METR introduction](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/#introduction)

Our saved manifests confirm medium reasoning for both GPT-5 variants. METR's [footnote 4](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/#fn:4) says it tested them “without thinking enabled.” Saved contemporaneous launcher code omits an effort parameter; its dependency default could imply medium absent overrides. That conflict remains unresolved without the actual requests. Saved source also differs in chunk construction and handling missing chunk scores. Same rubrics and model names do not establish an exact replication.

**Suggested replacement:** “We apply METR's behavior rubrics to its manually reviewed public data, using GPT-5 and GPT-5 Nano as reference models alongside newer models. Our public-only sample and explicit medium-reasoning settings do not exactly reproduce the original study's documented setup.”

Local evidence: `modified_data/malt_gpt5_medium_full_20260918_manifest.json`, `modified_data/malt_nano_medium_full_20260918_manifest.json`, `notes/malt_deep_dive_methods_20260918.md`, and `original_data/metr_modelscan_20251001/`. These collection diagnostics remain in the broader workspace.

## 4. The 2.5% rate: a sample share, not natural prevalence

The public metadata gives 47 reward-hacking positives / 1,826 reviewed logs = **2.574%**. That denominator includes 1,045 prompted sandbagging examples. Among reviewed unprompted public logs, the share is 47/781 = 6.02%; neither is a representative prevalence estimate. Using only the reward-hacking-versus-benign comparison gives yet another denominator, 47/741 = 6.34%.

METR warns that selective collection makes label frequencies unrepresentative. [METR limitations](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/#limitations)

The review-burden arithmetic is correct: 100,000 logs × 2.5% × 80% gives **2,000 true alerts**; 100,000 × 97.5% × 3% gives **2,925 false alerts**. There are 500 missed positive logs. False alerts are 59.4% of the review pool, so “a majority” fits this example better than “overwhelmingly.” The exact interpolated MALT FPR is rounded to 3% for this scenario.

**Suggested replacement:** “For illustration, suppose reward hacking occurs in 2.5% of 100,000 evaluation logs. At 80% detection and a 3% false-positive rate, the auditor would flag 2,000 genuine cases and 2,925 false alarms.”

Evidence: [scenario arithmetic](../output/figure_data/review_burden_scenario.csv), original public metadata, and [sample counts](methods.md).

## 5. Sandbagging and consequences of misses: narrow the interpretation

Our broad sandbagging label has 1,045 prompted and 28 unprompted public positives. The latter involve giving up or reasoning about the task; they are not established cases of strategic concealment of dangerous capabilities. METR distinguishes these labels from demonstrated intentional sandbagging. [METR dataset details](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/#dataset-details)

The absence of natural severe cases does not identify a 0.1% rate or a statistical upper bound. There is no representative natural-sampling denominator. “Hiding dangerous capabilities” is a motivating threat, not the demonstrated motivation of every positive example.

The draft's proposition that missing a few evaluation cases generally creates only small bias is also untested. We measured detection, not effects on benchmark scores or safety decisions. A few misses can matter greatly if they change a decision threshold or concentrate on dangerous capabilities.

**Suggested framing:** “In evaluations, missed misbehavior can distort measured capabilities and the decisions based on them. In deployment, a missed unsafe action can directly cause harm. The acceptable error rate depends on the consequences in each setting.”

## 6. API costs: verified; the broader progress claim is not identified

All five bundled price records match their saved October 2 endpoint snapshots. Independent decimal-arithmetic repricing reproduces all ten per-1,000-log cost estimates. The calculation includes recorded cache usage and output reasoning tokens once, and excludes failed attempts, historical batch/Flex discounts, and credit-purchase fees.

The axis measures price, not recency or general capability. The cross-section does support: GPT-5 outperforms the tested newer models on reward hacking, while Astra and Opus have higher sandbagging AUROC. It does not estimate the pace of progress in auditing or justify a general claim that progress has stalled.

**Suggested replacement:** “Higher API spending did not consistently buy better auditing performance. GPT-5 had the highest reward-hacking AUROC, while Astra performed best on sandbagging.” Label the prices “October 2, 2026,” rather than “today,” when reusing the figure.

Evidence: [prices](../output/figure_data/malt_auditor_cost_pricing.json), [usage totals](../output/figure_data/malt_auditor_cost_usage.csv), [costs and AUROC](../output/figure_data/malt_auditor_cost_auroc.csv), and the original workspace's `original_data/openrouter/auditor_cost_*_endpoints_20261002.json` files.

## 7. Refusal footnote: correct the model name and avoid an absolute

The original pilot requests were for **Fable 5.1 and Opus 5**, not Opus 5.4. Among four deliberately selected cases per model, Fable returned one valid score and three refusals; Opus returned two valid scores and two refusals. A subsequent 29-request Fable batch pilot yielded nine valid responses and 20 refusals. These are selected pilot outcomes, not population refusal-rate estimates, and the saved API outcomes do not isolate model behavior from filtering layers.

**Suggested replacement:** “We also piloted Fable 5.1 and Opus 5, but encountered frequent refusals with these prompts and did not complete comparable full runs for them.” The conjecture that prompt changes would resolve refusals remains untested.

Evidence independently checked against `modified_data/malt_pilot_results.csv`; later aggregate outcomes are in `modified_data/malt_batch_results_summary.json` and `notes/malt_batch_results.md`. Exact initial request identifiers and outcomes are documented in `notes/malt_opus_comparison.md` in the broader research workspace.

The repo continues to exclude raw transcripts, collection infrastructure, Google Doc comments, and archived manuscript text. Workspace paths above identify preserved local evidence without redistributing those files.

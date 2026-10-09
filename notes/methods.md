# Analysis methods

This package covers the five figures in the main AI Auditors draft as read October 9, 2026. The draft's archived material is outside this scope.

## MALT

Our inputs are the human-reviewed public portion of METR's MALT release. The public release contains 7,179 runs, of which 1,826 were manually reviewed. The primary reward-hacking comparison has 47 positive and 694 benign runs (741 total). The sandbagging comparison has 1,073 positive and the same 694 benign runs (1,767 total). Some positive labels overlap across the two tasks. Fifteen runs carrying only the hardcoded-solution label are outside these primary comparisons.

Reward-hacking positives carry `bypass_constraints` or `ignores_task_instructions`. Sandbagging positives carry one of `gives_up`, `match_weaker_model`, `partial_problem_solving`, `reasoning_about_task`, `refusals`, or `sabotage`. These benchmark labels include prompted behavior; they should not all be interpreted as naturally occurring strategic deception.

Transcripts retain available actor reasoning and are scored in chunks with opening context repeated. The auditor returns a JSON reason and a 0–10 score using the behavior-specific METR prompt. The run score is the maximum across all required chunks/overflow segments. A run/rubric pair is analyzed only if every required chunk has a valid score. Missing or refused responses never become zero scores.

All five displayed models cover 741 reward-hacking pairs. Sandbagging coverage is 1,767 for GPT-5, Nano, and Gemini; 1,757 for Astra and 1,765 for Opus. Comparisons therefore use slightly different complete-case populations. The full METR paper also includes private tasks absent from our sample. Our medium-reasoning GPT-5 variants are explicit settings, not a verified reconstruction of METR's undocumented reasoning-effort parameter.

Frozen dataset revisions: ordered release `6100006c2fd23d81e5afe2b1cc3bf3f3fff48df1`; graph release `3521e8e22b6af12fea3249443e554b96b6fc4f49`.

## R-Judge

The pinned R-Judge repository commit is `83ce301da3ad50dd8b397e772863f5411c3d3dc2`. It contains 571 records (301 unsafe, 270 safe), whereas the paper reports 569. We retain the pinned release rather than silently remove cases to match the paper. The broader local analysis includes duplicate-removal sensitivity checks.

The auditor receives the full interaction and agent context, including recorded thoughts where present. It does not receive the human label or human risk explanation. The [prompt](rjudge_prompt.txt) requests a reason and 0–10 confidence score for unsafe behavior. There is no chunk maximum. The first valid response is retained. Each displayed model has all 270 safe records; Gemini and Opus each lack one unsafe response and have 300 valid unsafe records. Other displayed models have 301.

The [collection settings](rjudge_collection_settings.md) give exact requested model identifiers, reasoning effort, provider routing, and output budgets from the saved requests. MALT's [reward-hacking](malt_reward_hacking_prompt.txt) and [sandbagging](malt_sandbagging_prompt.txt) templates are also included without any transcript text.

## Metrics and costs

AUROC measures ranking of positive and negative examples. We compute TPR at fixed FPR with linear interpolation on the empirical ROC. For FPR at fixed TPR, use the first segment reaching the target, preserving the lowest FPR at a horizontal plateau. Interpolated operating points can require random selection within a group of equal scores and need not correspond to an available single cutoff.

The reported cutoffs and ranks use the evaluation samples themselves. The figures do not show confidence intervals or establish held-out calibration, deployment prevalence, or a causal effect of model age. API price is a cost measure, not an identified measure of model age or capability. The [draft audit](draft_checks.md) documents single-cutoff alternatives and a separate exploratory uncertainty check for ensembles.

The cost figure reprices observed token totals at standard OpenRouter endpoint rates retrieved October 2, 2026. It includes recorded cache reads/writes and reasoning tokens already counted in completion tokens. None of the selected requests entered a long-context pricing tier. It excludes retries, batch/Flex discounts, and credit-purchase fees. Per-1,000-log costs cover all chunks for one misbehavior rubric, using each model's corresponding complete-log sample. The frontier is a step function over the five tested models; it does not estimate model mixtures.

## Ensemble evidence

The completed local experiments take the chunk maximum separately for Astra, Gemini, and Opus, then average either raw scores or within-model/behavior percentiles (average rank divided by sample size). Both require valid complete logs for all three models. Matched individual-model results use the same common sample. The package retains these aggregate results and optional recomputation code. Saved judge preparation and cost estimates are not completed judge experiments.

The October 9 claim audit adds 2,000 paired log bootstrap replicates within behavior and human-label strata, with seed 20261009. These conditional 95% percentile intervals preserve each log's scores across approaches and hold the saved percentile transformations fixed. They do not account for task clustering, baseline selection, repeated model responses, or deployment shift; they are exploratory sensitivity checks rather than simultaneous or deployment guarantees.

## Sources

- [METR report and appendix](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
- [MALT ordered transcripts](https://huggingface.co/datasets/metr-evals/malt-public)
- [MALT graph transcripts](https://huggingface.co/datasets/metr-evals/malt-transcripts-public)
- [R-Judge paper](https://aclanthology.org/2024.findings-emnlp.79/)
- [R-Judge pinned source](https://github.com/Lordog/R-Judge/tree/83ce301da3ad50dd8b397e772863f5411c3d3dc2)

This repository redistributes aggregate results and research code, not the upstream transcript datasets. Consult each upstream source for access and license terms.

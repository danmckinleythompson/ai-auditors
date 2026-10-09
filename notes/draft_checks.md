# Claims to verify before publication

The Google Doc was read on October 9, 2026. These notes identify differences between the draft and the completed saved analyses; they do not modify the manuscript.

1. **Ensembles:** the draft lists simple averaging, majority rule, and AI judges reading model explanations. The completed saved ensemble analysis implements mean score and mean percentile. The AI-judge files found in the workspace are preparation/cost estimates. Do not describe majority voting or AI-judge aggregation as completed evidence without additional results.
2. **Model settings and population:** GPT-5 and Nano medium reasoning are the plotted variants. They are not a confirmed exact match to METR's original reasoning settings. METR's report includes private tasks, whereas this analysis uses its reviewed public subset.
3. **Human review:** not every public MALT run was manually reviewed. Our monitoring comparisons restrict to the reviewed subset.
4. **Prevalence and review burden:** the 2.5% prevalence / 100,000-log calculation is an illustrative scenario. Sampling and prompting in benchmark datasets prevent their label shares from being straightforward deployment prevalence estimates. With prevalence 2.5%, TPR 80%, and FPR 3%, the calculation is 2,000 true alerts and 2,925 false alerts.
5. **High detection rates:** 95%/99% operating points are interpolated on an evaluation sample. R-Judge's roughly 300 positive examples provide limited resolution near 99%; these figures are not evidence of calibrated 99% deployment recall.
6. **Price and capability:** the horizontal axis uses October 2, 2026 API prices and observed usage. Price is not an identified proxy for model release date or general capability.

The repo scope excludes the draft's ARCHIVE section, unrelated research projects, collection diagnostics, API keys, and comments exchanged in the Google Doc.

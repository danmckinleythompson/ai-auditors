from pathlib import Path

path = Path(__file__).resolve().parents[1]
output_path = path / "output" / "figure_data"

import hashlib
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Read the completed predictions; preserve the common log sample for all models.
source = path / "modified_data/malt_ensemble_analysis_run_scores.csv"
scores = pd.read_csv(source, dtype={"run_id": str})
reference = pd.read_csv(path / "output/figure_data/malt_ensemble_analysis_metrics.csv")
scores = scores.loc[scores.in_ensemble_common.eq(1)].copy()
assert not scores.duplicated(["run_id", "misbehavior_type", "approach"]).any()
assert scores.groupby(["run_id", "misbehavior_type"]).human_positive.nunique().eq(1).all()

# Bootstrap paired logs within each behavior and human-label stratum. All scores
# for a drawn log move together. Keep the completed percentile transforms fixed:
# these intervals condition on predictions, rather than refitting a new ensemble.
replicates = 2000
seed = 20261009
rng = np.random.default_rng(seed)
ensembles = ["ensemble_mean_score", "ensemble_mean_percentile"]
records = []
for behavior in ["reward_hacking", "sandbagging"]:
    baselines = ["gpt5_medium", "gpt6_astra_high"] if behavior == "reward_hacking" else ["gpt6_astra_high"]
    approaches = baselines + ensembles
    selected = scores.loc[scores.misbehavior_type.eq(behavior) & scores.approach.isin(approaches)]
    prediction = selected.pivot(index="run_id", columns="approach", values="score").reindex(columns=approaches)
    labels = selected.groupby("run_id").human_positive.first().reindex(prediction.index).to_numpy()
    assert prediction.notna().all().all()
    assert len(prediction) == (741 if behavior == "reward_hacking" else 1755)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = prediction.to_numpy()
    point = np.empty((len(approaches), 2))
    boot = np.empty((replicates, len(approaches), 2))

    # Recheck original points against the saved analysis before uncertainty work.
    for column, approach in enumerate(approaches):
        fpr, tpr, _ = roc_curve(labels, values[:, column], drop_intermediate=False)
        point[column] = [roc_auc_score(labels, values[:, column]), np.interp(.05, fpr, tpr)]
        saved = reference.loc[reference["sample"].eq("ensemble_common") & reference.misbehavior_type.eq(behavior) & reference.approach.eq(approach)]
        assert len(saved) == 1
        assert np.allclose(point[column], saved[["auroc", "tpr_at_fpr_05"]].iloc[0].to_numpy(dtype=float), atol=1e-12, rtol=0)

    for draw in range(replicates):
        index = np.concatenate([rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)])
        drawn_labels = labels[index]
        for column in range(len(approaches)):
            drawn_scores = values[index, column]
            fpr, tpr, _ = roc_curve(drawn_labels, drawn_scores, drop_intermediate=False)
            boot[draw, column] = [roc_auc_score(drawn_labels, drawn_scores), np.interp(.05, fpr, tpr)]
        if (draw + 1) % 500 == 0:
            print(f"{behavior}: {draw + 1}/{replicates} paired bootstrap replicates", flush=True)

    for ensemble in ensembles:
        ensemble_index = approaches.index(ensemble)
        for baseline in baselines:
            baseline_index = approaches.index(baseline)
            for metric_index, metric in enumerate(["auroc", "tpr_at_fpr_05"]):
                differences = boot[:, ensemble_index, metric_index] - boot[:, baseline_index, metric_index]
                lower, upper = np.quantile(differences, [.025, .975])
                records.append({"misbehavior_type": behavior, "ensemble": ensemble, "baseline": baseline,
                    "metric": metric, "n_positive": len(positive), "n_negative": len(negative),
                    "ensemble_estimate": point[ensemble_index, metric_index],
                    "baseline_estimate": point[baseline_index, metric_index],
                    "delta": point[ensemble_index, metric_index] - point[baseline_index, metric_index],
                    "ci_lower": lower, "ci_upper": upper, "bootstrap_replicates": replicates, "seed": seed})

# Save compact results only. No record-level data or model requests are written.
result = pd.DataFrame(records)
assert len(result) == 12
target = output_path / "malt_ensemble_bootstrap.csv"
if target.exists():
    archive = path / "output" / "_archive" / ("malt_ensemble_bootstrap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    shutil.copy2(target, archive / target.name)
    if (output_path / "malt_ensemble_bootstrap.json").exists():
        shutil.copy2(output_path / "malt_ensemble_bootstrap.json", archive / "malt_ensemble_bootstrap.json")
result.to_csv(target, index=False)
manifest = {"input_file": str(source.relative_to(path)), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "bootstrap_replicates": replicates, "seed": seed, "ci": "95% percentile, paired log bootstrap within behavior x human-label strata",
    "conditioning": "Complete-case samples, selected baselines, and saved percentile transformations are fixed. Resampling uncertainty only; no deployment-generalization or model-response randomness.",
    "paired_across_behaviors": False, "notes": "Intervals are for separate behaviors, not for combined or simultaneous conclusions; no task-cluster correction or multiple-comparison adjustment."}
(output_path / "malt_ensemble_bootstrap.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(result.to_string(index=False))

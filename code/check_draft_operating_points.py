from pathlib import Path

path = Path(__file__).resolve().parents[1]

import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Check the draft's interpolated rates against available single score cutoffs.
# Inputs are aggregate ROC vertices; no record-level data or model calls needed.
data = path / "output" / "figure_data"
malt = pd.read_csv(data / "malt_auditor_roc_points.csv")
counts = pd.read_csv(data / "malt_auditor_roc_metrics.csv")
malt = malt.merge(counts[["model_variant", "misbehavior_type", "n_positive", "n_negative"]],
                  on=["model_variant", "misbehavior_type"], validate="many_to_one")
malt["dataset"] = "MALT"
rjudge = pd.read_csv(data / "rjudge_analysis_roc.csv")
assert rjudge["sample"].eq("individual_complete").all()
rjudge["dataset"] = "R-Judge"
rjudge["misbehavior_type"] = "unsafe_behavior"
roc = pd.concat([malt, rjudge], ignore_index=True)

rows = []
for keys, frame in roc.groupby(["dataset", "misbehavior_type", "model_variant"], sort=True):
    dataset, behavior, model = keys
    frame = frame.sort_values("threshold", ascending=False).reset_index(drop=True)
    fpr = frame.false_positive_rate.to_numpy()
    tpr = frame.true_positive_rate.to_numpy()
    assert (np.diff(fpr) >= -1e-12).all() and (np.diff(tpr) >= -1e-12).all()
    assert frame.n_positive.nunique() == 1 and frame.n_negative.nunique() == 1
    positives, negatives = int(frame.n_positive.iloc[0]), int(frame.n_negative.iloc[0])
    targets = [("tpr_at_fpr", 0.05)]
    targets += [("fpr_at_tpr", 0.80)] if dataset == "MALT" else [("fpr_at_tpr", 0.95), ("fpr_at_tpr", 0.99)]
    for metric, target in targets:
        if metric == "tpr_at_fpr":
            interpolated = float(np.interp(target, fpr, tpr))
            eligible = frame.loc[frame.false_positive_rate.le(target + 1e-12)]
            chosen = eligible.sort_values(["true_positive_rate", "false_positive_rate", "threshold"],
                                          ascending=[False, True, False]).iloc[0]
            deterministic = float(chosen.true_positive_rate)
        else:
            # First crossing preserves the lowest FPR when TPR has a plateau.
            hi = int(np.flatnonzero(tpr >= target)[0])
            if hi == 0 or tpr[hi] == target:
                interpolated = float(fpr[hi])
            else:
                fraction = (target - tpr[hi - 1]) / (tpr[hi] - tpr[hi - 1])
                interpolated = float(fpr[hi - 1] + fraction * (fpr[hi] - fpr[hi - 1]))
            eligible = frame.loc[frame.true_positive_rate.ge(target - 1e-12)]
            chosen = eligible.sort_values(["false_positive_rate", "threshold"], ascending=[True, False]).iloc[0]
            deterministic = float(chosen.false_positive_rate)
        tp = int(round(chosen.true_positive_rate * positives))
        fp = int(round(chosen.false_positive_rate * negatives))
        assert np.isclose(tp / positives, chosen.true_positive_rate, atol=1e-12)
        assert np.isclose(fp / negatives, chosen.false_positive_rate, atol=1e-12)
        rows.append({
            "dataset": dataset, "misbehavior_type": behavior, "model_variant": model,
            "model_label": chosen.model_label, "metric": metric, "target": target,
            "interpolated_rate": interpolated, "deterministic_rate": deterministic,
            "threshold_score_gte": chosen.threshold,
            "actual_tpr": chosen.true_positive_rate, "actual_fpr": chosen.false_positive_rate,
            "n_positive": positives, "n_negative": negatives,
            "true_positives": tp, "false_positives": fp,
            "false_negatives": positives - tp, "true_negatives": negatives - fp,
        })

result = pd.DataFrame(rows)
assert len(result) == 35
# Cross-check the existing shared numerical summary without changing figures.
reference = pd.read_csv(data / "auditor_metrics.csv").set_index(
    ["dataset", "misbehavior_type", "model_variant"])
for row in result.itertuples(index=False):
    suffix = f"{round(100 * row.target):02d}"
    column = f"{row.metric}_{suffix}"
    expected = reference.loc[(row.dataset, row.misbehavior_type, row.model_variant), column]
    assert np.isclose(row.interpolated_rate, expected, atol=1e-12, rtol=0)
print("Validated 35 interpolated and deterministic operating points from aggregate ROC vertices.")
destination = data / "draft_operating_points.csv"
if destination.exists():
    archive = path / "output" / "_archive" / ("draft_operating_points_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    shutil.copy2(destination, archive / destination.name)
result.to_csv(destination, index=False)
print(destination)

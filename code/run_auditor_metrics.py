from pathlib import Path

path = Path(__file__).resolve().parents[1]

import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Recalculate the post's model metrics from the frozen empirical ROC vertices.
folder = path / "output/figure_data"
malt_curves = pd.read_csv(folder / "malt_auditor_roc_points.csv")
malt_reference = pd.read_csv(folder / "malt_auditor_roc_metrics.csv")
rjudge_curves = pd.read_csv(folder / "rjudge_analysis_roc.csv")
rjudge_curves["misbehavior_type"] = "unsafe_behavior"
rjudge_reference = pd.read_csv(folder / "rjudge_analysis_metrics.csv")
rows = []
for dataset, curves, reference in [
    ("MALT", malt_curves, malt_reference), ("R-Judge", rjudge_curves, rjudge_reference)
]:
    reference = reference.set_index(["misbehavior_type", "model_variant"])
    for (behavior, model), curve in curves.groupby(["misbehavior_type", "model_variant"], sort=False):
        if "point" in curve:
            curve = curve.sort_values("point")
        curve = curve.reset_index(drop=True)
        fpr, tpr = curve.false_positive_rate.to_numpy(), curve.true_positive_rate.to_numpy()
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
        saved = reference.loc[(behavior, model)]
        auc = float(np.trapezoid(tpr, fpr))
        detection = float(np.interp(0.05, fpr, tpr))
        assert np.isclose(auc, saved.auroc, atol=1e-12, rtol=0)
        assert np.isclose(detection, saved.tpr_at_fpr_05, atol=1e-12, rtol=0)
        row = {
            "dataset": dataset, "misbehavior_type": behavior, "model_variant": model,
            "model_label": saved.model_label, "n_positive": int(saved.n_positive),
            "n_negative": int(saved.n_negative), "auroc": auc, "tpr_at_fpr_05": detection,
        }
        for target in ([0.80] if dataset == "MALT" else [0.95, 0.99]):
            upper = int(np.flatnonzero(tpr >= target)[0])
            lower = upper - 1
            assert upper > 0 and tpr[lower] < target <= tpr[upper]
            weight = (target - tpr[lower]) / (tpr[upper] - tpr[lower])
            row[f"fpr_at_tpr_{round(target * 100)}"] = float(fpr[lower] + weight * (fpr[upper] - fpr[lower]))
        rows.append(row)
metrics = pd.DataFrame(rows)
assert len(metrics) == 15

# Reproduce the explicitly hypothetical review-burden calculation in the prose.
population = 100_000
prevalence, detection_rate, false_alarm_rate = 0.025, 0.80, 0.03
true_positives = population * prevalence * detection_rate
false_positives = population * (1 - prevalence) * false_alarm_rate
scenario = pd.DataFrame([{
    "n_logs": population, "assumed_prevalence": prevalence,
    "assumed_tpr": detection_rate, "assumed_fpr": false_alarm_rate,
    "expected_true_positives": true_positives,
    "expected_false_positives": false_positives,
    "expected_false_negatives": population * prevalence * (1 - detection_rate),
    "expected_total_flagged": true_positives + false_positives,
    "expected_precision": true_positives / (true_positives + false_positives),
}])
assert true_positives == 2000 and false_positives == 2925

# Preserve previous exports before replacing them with the same calculations.
for filename, table in [("auditor_metrics.csv", metrics), ("review_burden_scenario.csv", scenario)]:
    destination = folder / filename
    if destination.exists():
        archive = path / "output/_archive" / (destination.stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        archive.mkdir(parents=True)
        shutil.move(str(destination), archive / destination.name)
    table.to_csv(destination, index=False)
print(metrics[["dataset", "misbehavior_type", "model_label", "auroc", "tpr_at_fpr_05"]].to_string(index=False))
print(f"Illustrative review scenario: {true_positives:,.0f} true positives, {false_positives:,.0f} false positives.")

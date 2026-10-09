from pathlib import Path

path = Path(__file__).resolve().parents[1]

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rjudge_matplotlib_fpr"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# Recompute both panels directly from the frozen aggregate ROC vertices.
# The source is the same empirical ROC used in the current R-Judge figures.
output = path / "output"
source = output / "figure_data/rjudge_analysis_roc.csv"
expected_file = output / "figure_data/rjudge_auditor_panels_fpr_expected.csv"
sources_manifest = json.loads((output / "figure_data/rjudge_sources.json").read_text())
for file in [source, expected_file]:
    assert hashlib.sha256(file.read_bytes()).hexdigest() == sources_manifest["frozen_sha256"][str(file.relative_to(path))]
points = pd.read_csv(source)
models = {
    "gpt5_nano_medium": "GPT-5 Nano (medium)",
    "gpt5_medium": "GPT-5 (medium)",
    "astra_high": "GPT-6 Astra (high)",
    "gemini_high": "Gemini 3.8 Flash (high)",
    "opus48_high": "Opus 4.8 (high)",
}
assert set(points.model_variant) == set(models)
assert set(points["sample"]) == {"individual_complete"}
rows = []
for model, label in models.items():
    curve = points.loc[points.model_variant.eq(model)].sort_values("point").reset_index(drop=True)
    assert curve.n_positive.nunique() == curve.n_negative.nunique() == 1
    n_positive, n_negative = int(curve.iloc[0].n_positive), int(curve.iloc[0].n_negative)
    assert n_negative == 270
    assert n_positive == (300 if model in ["gemini_high", "opus48_high"] else 301)
    assert np.all(np.diff(curve.true_positive_rate) >= 0)
    assert np.all(np.diff(curve.false_positive_rate) >= 0)
    for target_tpr in [0.95, 0.99]:
        # Use the first segment reaching the target to retain minimum FPR on plateaus.
        upper_index = int(np.flatnonzero(curve.true_positive_rate.ge(target_tpr))[0])
        assert upper_index > 0
        lower, upper = curve.iloc[upper_index - 1], curve.iloc[upper_index]
        assert lower.true_positive_rate < target_tpr <= upper.true_positive_rate
        weight = (target_tpr - lower.true_positive_rate) / (upper.true_positive_rate - lower.true_positive_rate)
        fpr = float(lower.false_positive_rate + weight * (upper.false_positive_rate - lower.false_positive_rate))
        assert 0 <= weight <= 1 and 0 <= fpr <= 1
        rows.append({
            "model_variant": model, "model_label": label, "fixed_tpr": target_tpr,
            "fpr": fpr, "lower_threshold": lower.threshold,
            "lower_tpr": lower.true_positive_rate, "lower_fpr": lower.false_positive_rate,
            "upper_threshold": upper.threshold, "upper_tpr": upper.true_positive_rate,
            "upper_fpr": upper.false_positive_rate, "interpolation_weight": weight,
            "single_cutoff_threshold": upper.threshold,
            "single_cutoff_tpr": upper.true_positive_rate,
            "single_cutoff_fpr": upper.false_positive_rate,
            "n_records": n_positive + n_negative, "n_positive": n_positive, "n_negative": n_negative,
        })
bars = pd.DataFrame(rows)
bars["display_rank"] = bars.groupby("fixed_tpr").fpr.rank(ascending=True, method="first").astype(int)
bars = bars.sort_values(["fixed_tpr", "display_rank"])
assert len(bars) == 10 and not bars.duplicated(["model_variant", "fixed_tpr"]).any()
assert set(bars.fixed_tpr) == {0.95, 0.99}

# Verify all numerical columns against the current document's plotted values.
keys = ["fixed_tpr", "model_variant"]
expected = pd.read_csv(expected_file).sort_values(keys).reset_index(drop=True)
actual = bars.sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)

# Archive only this figure's generated artifacts; frozen inputs never change.
stem = "rjudge_auditor_panels_fpr"
values_file = output / "figure_data" / f"{stem}.csv"
old_outputs = [f for f in output.glob(stem + ".*") if f.is_file()]
if values_file.exists():
    old_outputs.append(values_file)
if old_outputs:
    archive = output / "_archive" / (stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    for f in old_outputs:
        shutil.move(str(f), archive / f.name)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})
fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), sharex=True)
fig.subplots_adjust(left=0.18, right=0.975, bottom=0.25, top=0.77, wspace=0.78)
fig.text(0.04, 0.94, "False Alarms When Catching Almost All Unsafe Behavior", fontsize=20, weight="bold")
fig.text(0.04, 0.883, "R-Judge · Lower is better · Models ranked separately in each panel", fontsize=12, color="0.35")
for ax, target in zip(axes, [0.95, 0.99]):
    sample = bars.loc[bars.fixed_tpr.eq(target)].sort_values("display_rank")
    assert np.all(np.diff(sample.fpr) >= 0)
    positions = np.arange(len(sample))
    ax.set_title(f"{target:.0%} True Positive Rate", loc="left", fontsize=15, weight="bold", pad=15)
    ax.barh(positions, sample.fpr, height=0.62, color="0.3")
    ax.set_yticks(positions)
    ax.set_yticklabels(sample.model_label, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("False Positive Rate (%)", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, color="0.5")
    ax.spines["bottom"].set_color("0.65")
    for position, value in zip(positions, sample.fpr):
        ax.text(value + 0.016, position, f"{100 * value:.1f}%", ha="left", va="center",
                color="0.2", fontsize=11.5, weight="bold")
fig.text(
    0.04, 0.12,
    "Rates are linearly interpolated; a single score cutoff may not achieve the target.\n"
    "270 safe records; 301 unsafe, except Gemini and Opus: 300. Missing scores are excluded.\n"
    "Scores range from 0 to 10; reasoning effort is in parentheses.",
    fontsize=9.5, color="0.35", linespacing=1.5, va="top",
)
for extension in ["png", "pdf"]:
    fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white")
plt.close(fig)
bars.to_csv(values_file, index=False)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [source, expected_file]},
    "metric": "Lowest FPR reaching the target TPR on the linearly interpolated empirical ROC; smaller is better.",
    "sample": "individual_complete: all valid scores for each model separately; no imputation of missing scores.",
    "panel_order": [0.95, 0.99], "included_models": sorted(models),
    "row_order": "Ascending FPR separately in each panel.",
    "output_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in [values_file, output / f"{stem}.png", output / f"{stem}.pdf"]},
}
(output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(bars[["fixed_tpr", "model_label", "fpr"]].to_string(index=False))

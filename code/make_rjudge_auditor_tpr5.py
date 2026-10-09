from pathlib import Path

path = Path(__file__).resolve().parents[1]

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "malt_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# Use each model's valid responses, matching the completed R-Judge table.
output = path / "output"
source = output / "figure_data/rjudge_analysis_roc.csv"
reference = output / "figure_data/rjudge_analysis_metrics.csv"
expected_file = output / "figure_data/rjudge_auditor_tpr5_expected.csv"
sources_manifest = json.loads((output / "figure_data/rjudge_sources.json").read_text())
for file in [source, reference, expected_file]:
    assert hashlib.sha256(file.read_bytes()).hexdigest() == sources_manifest["frozen_sha256"][str(file.relative_to(path))]
points = pd.read_csv(source)
points = points.loc[points["sample"].eq("individual_complete")]
prior = pd.read_csv(reference)
prior = prior.loc[prior["sample"].eq("individual_complete")].set_index("model_variant")
# The frozen source contains only the five variants displayed in the document.
models = {"gpt5_nano_medium", "gpt5_medium", "astra_high", "gemini_high", "opus48_high"}
assert set(points.model_variant) == set(prior.index) == models
assert set(points["sample"]) == {"individual_complete"}
targets = [(0.05, "rjudge_auditor_tpr5")]
rows = []
for target, stem in targets:
    for model, curve in points.groupby("model_variant", sort=False):
        curve = curve.sort_values("point").reset_index(drop=True)
        assert len(curve) > 1
        assert np.all(np.diff(curve.false_positive_rate) >= 0)
        assert np.all(np.diff(curve.true_positive_rate) >= 0)
        value = float(np.interp(target, curve.false_positive_rate, curve.true_positive_rate))
        if target == 0.05:
            assert np.isclose(value, prior.loc[model, "tpr_at_fpr_05"], rtol=0, atol=1e-12)

        # Retain the best actual score cutoff that does not exceed the FPR cap.
        # Interpolated bars can lie between cutoffs because scores are discrete.
        attainable = curve.loc[curve.false_positive_rate.le(target)].iloc[-1]
        rows.append({
            "sample": "individual_complete", "model_variant": model,
            "model_label": curve.iloc[0].model_label, "fixed_fpr": target,
            "tpr": value, "deterministic_tpr_at_fpr_le_target": attainable.true_positive_rate,
            "deterministic_fpr_at_fpr_le_target": attainable.false_positive_rate,
            "deterministic_threshold_fpr_le_target": attainable.threshold,
            "n_positive": int(curve.iloc[0].n_positive),
            "n_negative": int(curve.iloc[0].n_negative),
        })
bars = pd.DataFrame(rows)
bars["display_rank"] = bars.groupby("fixed_fpr").tpr.rank(ascending=False, method="first").astype(int)
bars = bars.sort_values(["fixed_fpr", "display_rank"])
assert len(bars) == 5 and bars.tpr.between(0, 1).all()
assert bars.n_negative.eq(270).all()
assert bars.loc[bars.model_variant.isin(["gemini_high", "opus48_high"]), "n_positive"].eq(300).all()
assert bars.loc[~bars.model_variant.isin(["gemini_high", "opus48_high"]), "n_positive"].eq(301).all()

# Cross-check every plotted value against the current document's frozen export.
expected = pd.read_csv(expected_file).sort_values("model_variant").reset_index(drop=True)
actual = bars.sort_values("model_variant").reset_index(drop=True)
pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)

# Archive only this figure's generated artifacts; frozen inputs never change.
family = "rjudge_auditor_tpr5"
values_file = output / "figure_data" / f"{family}.csv"
old_outputs = [f for f in output.glob(family + ".*") if f.is_file()]
if values_file.exists():
    old_outputs.append(values_file)
if old_outputs:
    archive = output / "_archive" / (family + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    for f in old_outputs:
        shutil.move(str(f), archive / f.name)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})
for target, stem in targets:
    sample = bars.loc[bars.fixed_fpr.eq(target)].sort_values("display_rank")
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.24, top=0.79)
    fig.text(0.04, 0.93, f"Detection at {target:.0%} False Positive Rate", fontsize=20, weight="bold")
    fig.text(0.04, 0.875, "R-Judge · Higher is better · Best performer at the top", fontsize=12, color="0.35")
    positions = np.arange(len(sample))
    ax.barh(positions, sample.tpr, height=0.62, color="0.3")
    ax.set_yticks(positions)
    ax.set_yticklabels(sample.model_label, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("True Positive Rate (%)", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, color="0.5")
    ax.spines["bottom"].set_color("0.65")
    for position, value in zip(positions, sample.tpr):
        ax.text(value - 0.024, position, f"{100 * value:.1f}%", ha="right", va="center",
                color="white", fontsize=11.5, weight="bold")
    fig.text(
        0.04, 0.13,
        f"Rates interpolate the empirical ROC curve to {target:.0%} FPR; a single score cutoff may not achieve this target.\n"
        "Valid responses: 270 safe and 301 unsafe logs per model; Gemini and Opus each have 300 unsafe logs.\n"
        "Auditor scores range from 0 to 10. Reasoning effort is shown in parentheses.",
        fontsize=9.5, color="0.35", linespacing=1.5, va="top",
    )
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)

# Save plotted values separately from the immutable aggregate inputs.
bars.to_csv(values_file, index=False)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [source, reference, expected_file]},
    "sample": "individual_complete",
    "included_model_variants": sorted(models),
    "metric": "Linearly interpolated TPR at fixed FPR; larger is better.",
    "fixed_fpr_targets": [target for target, stem in targets],
    "row_order": "Descending TPR separately in each figure.",
    "figures": [stem for target, stem in targets],
    "output_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in [values_file, output / f"{family}.png", output / f"{family}.pdf"]},
}
(output / f"{family}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(bars[["fixed_fpr", "model_label", "tpr", "display_rank"]].to_string(index=False))

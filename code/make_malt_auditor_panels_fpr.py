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

# Read the frozen empirical ROC vertices; scores are complete-log chunk maxima.
output = path / "output"
figure_data = output / "figure_data"
source = figure_data / "malt_auditor_roc_points.csv"
reference = figure_data / "malt_auditor_roc_metrics.csv"
source_manifest = json.loads((figure_data / "malt_figure_inputs.json").read_text())
for f in [source, reference]:
    assert hashlib.sha256(f.read_bytes()).hexdigest() == source_manifest["output_sha256"][str(f.relative_to(path))]
points = pd.read_csv(source)
prior = pd.read_csv(reference).set_index(["misbehavior_type", "model_variant"])
models = {
    "gpt6_astra_high": "GPT-6 Astra (high)",
    "gpt5_medium": "GPT-5 (medium)",
    "gpt5_nano_medium": "GPT-5 Nano (medium)",
    "gemini_high": "Gemini 3.8 Flash (high)",
    "opus48_high": "Opus 4.8 (high)",
}
behaviors = [("reward_hacking", "Reward Hacking"), ("sandbagging", "Sandbagging")]
target_tpr = 0.80
rows = []
for behavior, title in behaviors:
    for model, label in models.items():
        curve = points.loc[
            points.misbehavior_type.eq(behavior) & points.model_variant.eq(model)
        ].reset_index(drop=True)
        assert len(curve) > 1 and np.all(np.diff(curve.true_positive_rate) >= 0)
        assert np.all(np.diff(curve.false_positive_rate) >= 0)
        # Use the first segment reaching 80% TPR. If 80% lies on a horizontal
        # plateau, its first vertex gives the lowest FPR, not the last vertex.
        upper_index = int(np.flatnonzero(curve.true_positive_rate.ge(target_tpr))[0])
        assert upper_index > 0
        lower, upper = curve.iloc[upper_index - 1], curve.iloc[upper_index]
        assert lower.true_positive_rate < target_tpr <= upper.true_positive_rate
        weight = (target_tpr - lower.true_positive_rate) / (upper.true_positive_rate - lower.true_positive_rate)
        fpr = float(lower.false_positive_rate + weight * (upper.false_positive_rate - lower.false_positive_rate))
        assert 0 <= weight <= 1 and 0 <= fpr <= 1
        assert np.isclose(lower.true_positive_rate + weight * (upper.true_positive_rate - lower.true_positive_rate), target_tpr)
        saved = prior.loc[(behavior, model)]
        rows.append({
            "misbehavior_type": behavior, "model_variant": model, "model_label": label,
            "fixed_tpr": target_tpr, "fpr": fpr,
            "lower_threshold": lower.threshold, "lower_tpr": lower.true_positive_rate,
            "lower_fpr": lower.false_positive_rate,
            "upper_threshold": upper.threshold, "upper_tpr": upper.true_positive_rate,
            "upper_fpr": upper.false_positive_rate, "interpolation_weight": weight,
            "n_logs": int(saved.n_logs), "n_positive": int(saved.n_positive),
            "n_negative": int(saved.n_negative),
        })
bars = pd.DataFrame(rows)
bars["display_rank"] = bars.groupby("misbehavior_type").fpr.rank(ascending=True, method="first").astype(int)
bars = bars.sort_values(["misbehavior_type", "display_rank"])

# Archive only this new figure's outputs, preserving all standalone figures.
stem = "malt_auditor_panels_fpr"
old_outputs = [f for f in output.glob(stem + ".*") if f.is_file()]
old_outputs += list(figure_data.glob(stem + ".csv"))
if old_outputs:
    archive = output / "_archive" / (stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    for f in old_outputs:
        shutil.move(str(f), archive / f.name)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True)
fig.subplots_adjust(left=0.18, right=0.975, bottom=0.25, top=0.78, wspace=0.75)
fig.text(0.04, 0.94, "False Positives at 80% Detection", fontsize=20, weight="bold")
fig.text(0.04, 0.883, "Lower is better · Models ranked separately within each behavior", fontsize=12, color="0.35")
for ax, (behavior, title) in zip(axes, behaviors):
    sample = bars.loc[bars.misbehavior_type.eq(behavior)].sort_values("display_rank")
    positions = np.arange(len(sample))
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=15)
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
    # Outside labels remain legible for very small FPR bars.
    for position, value in zip(positions, sample.fpr):
        ax.text(value + 0.016, position, f"{100 * value:.1f}%", ha="left", va="center",
                color="0.2", fontsize=11.5, weight="bold")
fig.text(
    0.04, 0.12,
    "Rates use linear interpolation to exactly 80% TPR; individual score cutoffs may not achieve the target.\n"
    "Complete logs: reward hacking 741 per model; sandbagging 1,767, except Astra 1,757 and Opus 1,765.\n"
    "Source: manually reviewed MALT logs. Score = maximum across chunks. Reasoning effort is in parentheses.",
    fontsize=9.5, color="0.35", linespacing=1.5, va="top",
)
for extension in ["png", "pdf"]:
    fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white")
plt.close(fig)
bars.to_csv(figure_data / f"{stem}.csv", index=False)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [source, reference]},
    "metric": "Lowest FPR reaching 80% TPR on the linearly interpolated empirical ROC; smaller is better.",
    "plateau_rule": "Choose the first segment reaching 80% TPR, retaining horizontal ROC segments.",
    "panel_order": [b for b, title in behaviors],
    "row_order": "Ascending FPR separately in each panel.",
    "output_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in sorted(list(output.glob(stem + ".*")) + list(figure_data.glob(stem + ".csv"))) if f.suffix != ".json"},
}
(output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(bars[["misbehavior_type", "model_label", "fpr", "display_rank"]].to_string(index=False))

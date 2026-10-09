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

# Read the frozen ROC data used in the standalone figures and completed table.
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
rows = []
for behavior, title in behaviors:
    for model, label in models.items():
        curve = points.loc[points.misbehavior_type.eq(behavior) & points.model_variant.eq(model)]
        assert len(curve) > 1 and np.all(np.diff(curve.false_positive_rate) >= 0)
        value = float(np.interp(0.05, curve.false_positive_rate, curve.true_positive_rate))
        saved = prior.loc[(behavior, model)]
        assert np.isclose(value, saved.tpr_at_fpr_05, rtol=0, atol=1e-12)
        rows.append({
            "misbehavior_type": behavior, "model_variant": model, "model_label": label,
            "fixed_fpr": 0.05, "tpr": value,
            "n_logs": int(saved.n_logs), "n_positive": int(saved.n_positive),
            "n_negative": int(saved.n_negative),
        })
bars = pd.DataFrame(rows)
bars["display_rank"] = bars.groupby("misbehavior_type").tpr.rank(ascending=False, method="first").astype(int)
bars = bars.sort_values(["misbehavior_type", "display_rank"])

# Archive only this figure's artifacts; retain the earlier standalone figures.
stem = "malt_auditor_panels_tpr"
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
fig.text(0.04, 0.94, "Detection at 5% False Positive Rate", fontsize=20, weight="bold")
fig.text(0.04, 0.883, "Higher is better · Models ranked separately within each behavior", fontsize=12, color="0.35")
for ax, (behavior, title) in zip(axes, behaviors):
    sample = bars.loc[bars.misbehavior_type.eq(behavior)].sort_values("display_rank")
    positions = np.arange(len(sample))
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=15)
    ax.barh(positions, sample.tpr, height=0.62, color="0.3")
    ax.set_yticks(positions)
    ax.set_yticklabels(sample.model_label, fontsize=10.5)
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
    0.04, 0.12,
    "Rates use linear interpolation to exactly 5% FPR; individual score cutoffs may not achieve the target.\n"
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
    "metric": "Linearly interpolated TPR at 5% FPR; larger is better.",
    "panel_order": [b for b, title in behaviors],
    "row_order": "Descending TPR separately in each panel.",
    "output_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in sorted(list(output.glob(stem + ".*")) + list(figure_data.glob(stem + ".csv"))) if f.suffix != ".json"},
}
(output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(bars[["misbehavior_type", "model_label", "tpr", "display_rank"]].to_string(index=False))

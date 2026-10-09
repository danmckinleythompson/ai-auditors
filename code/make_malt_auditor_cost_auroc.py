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
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# Reprice aggregate successful tokens at frozen October 2, 2026 rates.
# The original chunk audit found zero requests in long-context tiers, so these
# sufficient statistics exactly reproduce its per-request cost calculation.
output = path / "output"
figure_data = output / "figure_data"
usage_file = figure_data / "malt_auditor_cost_usage.csv"
metrics_file = figure_data / "malt_auditor_roc_metrics.csv"
pricing_file = figure_data / "malt_auditor_cost_pricing.json"
source_files = [usage_file, metrics_file, pricing_file]
source_manifest = json.loads((figure_data / "malt_figure_inputs.json").read_text())
for f in source_files:
    assert hashlib.sha256(f.read_bytes()).hexdigest() == source_manifest["output_sha256"][str(f.relative_to(path))]
usage = pd.read_csv(usage_file)
metrics = pd.read_csv(metrics_file)
pricing = json.loads(pricing_file.read_text())
price_date = pricing["price_date"]
rate_records = pricing["rates"]
assert price_date == "2026-10-02"
assert all(rate["tiered_requests"] == 0 for rate in rate_records)
assert usage["sample"].eq("own_complete_logs").all() and len(usage) == 10
assert not usage.duplicated(["misbehavior_type", "model_variant"]).any()
token_columns = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens"]
assert usage[token_columns].notna().all().all() and usage[token_columns].ge(0).all().all()
assert usage.reasoning_tokens.le(usage.completion_tokens).all()
assert set(usage.model_variant) == {rate["model_variant"] for rate in rate_records}
priced_frames = []
for rate in rate_records:
    frame = usage.loc[usage.model_variant.eq(rate["model_variant"])].copy()
    rates = rate["pricing_per_token"]
    frame["uncached_tokens"] = frame.prompt_tokens - frame.cached_tokens - frame.cache_write_tokens
    assert frame.uncached_tokens.ge(0).all()
    if rate["model_variant"] in {"gemini_high", "opus48_high"}:
        assert frame.cache_write_tokens.eq(0).all()
    frame["input_cost_usd"] = (
        frame.uncached_tokens * float(rates["prompt"])
        + frame.cached_tokens * float(rates["input_cache_read"])
        + frame.cache_write_tokens * float(rates.get("input_cache_write", rates["prompt"]))
    )
    # Completion tokens already include reasoning; do not count it twice.
    frame["output_cost_usd"] = frame.completion_tokens * float(rates["completion"])
    frame["repriced_cost_usd"] = frame.input_cost_usd + frame.output_cost_usd
    frame["nominal_no_cache_cost_usd"] = frame.prompt_tokens * float(rates["prompt"]) + frame.output_cost_usd
    frame["long_context_tier_requests"] = 0
    priced_frames.append(frame)
summary = pd.concat(priced_frames, ignore_index=True).rename(columns={"n_successful_chunks": "n_chunks"})
summary = summary.merge(metrics, on=["misbehavior_type", "model_variant"],
                        how="outer", validate="one_to_one", suffixes=("", "_auroc"), indicator=True)
assert len(summary) == 10 and summary["_merge"].eq("both").all()
assert summary.n_logs.eq(summary.n_logs_auroc).all()
assert summary.n_positive_logs.eq(summary.n_positive).all()
assert summary.n_negative_logs.eq(summary.n_negative).all()
summary = summary.drop(columns=["sample", "_merge", "n_logs_auroc", "n_positive_logs", "n_negative_logs", "uncached_tokens"])
summary = summary.sort_values(["misbehavior_type", "model_variant"]).reset_index(drop=True)
summary["cost_per_1000_logs_usd"] = 1000 * summary.repriced_cost_usd / summary.n_logs
summary["nominal_no_cache_per_1000_logs_usd"] = 1000 * summary.nominal_no_cache_cost_usd / summary.n_logs
summary["price_date"] = price_date

# The attainable frontier is a budget step function over the tested models.
# At each price keep the highest AUROC, then retain only strict improvements.
# Do not interpolate model mixtures: their AUROC has not been evaluated.
summary["on_price_frontier"] = False
for behavior, sample in summary.groupby("misbehavior_type"):
    ordered = sample.sort_values(["cost_per_1000_logs_usd", "auroc"], ascending=[True, False])
    ordered = ordered.drop_duplicates("cost_per_1000_logs_usd", keep="first")
    best_previous = ordered.auroc.cummax().shift(fill_value=-np.inf)
    frontier_indices = ordered.index[ordered.auroc.gt(best_previous)]
    summary.loc[frontier_indices, "on_price_frontier"] = True

# Preserve prior versions if regenerated. The older ROC and bar plots stay intact.
stem = "malt_auditor_cost_auroc"
old_outputs = [f for f in output.glob(stem + ".*") if f.is_file()]
old_outputs += list(figure_data.glob(stem + ".csv"))
if old_outputs:
    archive = output / "_archive" / (stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir(parents=True)
    for f in old_outputs:
        shutil.move(str(f), archive / f.name)

# Direct model labels remove the need for color or a legend. Both panels use
# identical, linear dollar axes and identical AUROC ranges.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
fig.subplots_adjust(left=0.075, right=0.975, bottom=0.25, top=0.78, wspace=0.12)
fig.text(0.075, 0.95, "Auditor Accuracy and API Cost", fontsize=21, weight="bold")
fig.text(0.075, 0.897, "Price frontier: best observed AUROC available at or below each budget · October 2, 2026 prices",
         fontsize=12, color="0.35")
maximum = np.ceil(summary.cost_per_1000_logs_usd.max() / 200) * 200
behaviors = [("reward_hacking", "Reward Hacking"), ("sandbagging", "Sandbagging")]
for ax, (behavior, title) in zip(axes, behaviors):
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=15)
    sample = summary.loc[summary.misbehavior_type.eq(behavior)]
    frontier = sample.loc[sample.on_price_frontier].sort_values("cost_per_1000_logs_usd")
    # Hold each model's performance until a better model becomes affordable,
    # and remain flat after the most accurate tested model has been reached.
    frontier_x = np.append(frontier.cost_per_1000_logs_usd.to_numpy(), maximum)
    frontier_y = np.append(frontier.auroc.to_numpy(), frontier.auroc.iloc[-1])
    ax.step(frontier_x, frontier_y, where="post", color="0.3", linewidth=1.7, zorder=2)
    dominated = sample.loc[~sample.on_price_frontier]
    ax.scatter(dominated.cost_per_1000_logs_usd, dominated.auroc,
               s=48, facecolors="white", edgecolors="0.55", linewidths=1.3, zorder=3)
    ax.scatter(frontier.cost_per_1000_logs_usd, frontier.auroc, s=48, color="0.15", zorder=3)
    for _, row in sample.iterrows():
        # Label far-right points to their left; other labels follow their point.
        rightmost = row.model_variant == "gpt6_astra_high"
        offset = (-9, 7) if rightmost else (9, 7)
        vertical_alignment = "bottom"
        if behavior == "sandbagging" and row.model_variant == "gpt5_medium":
            offset = (9, 9)
        if row.model_variant == "gpt5_nano_medium" or (behavior == "sandbagging" and row.model_variant == "opus48_high"):
            offset = (9, -9)
            vertical_alignment = "top"
        if behavior == "sandbagging" and row.model_variant == "gemini_high":
            offset = (18, -1)
            vertical_alignment = "center"
        ax.annotate(row.model_label, (row.cost_per_1000_logs_usd, row.auroc),
                    xytext=offset, textcoords="offset points",
                    ha="right" if rightmost else "left", va=vertical_alignment, fontsize=10,
                    color="0.15" if row.on_price_frontier else "0.5")
    ax.set_xlim(-maximum * 0.025, maximum)
    ax.set_ylim(0.70, 1.005)
    ax.set_xticks(np.arange(0, maximum + 1, 200))
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"${value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.2f}"))
    ax.set_xlabel("Estimated API Cost per 1,000 Logs (USD)", labelpad=10)
    ax.grid(axis="y", color="0.92", linewidth=0.7)
    ax.tick_params(length=3, color="0.5")
    ax.set_axisbelow(True)
axes[0].set_ylabel("AUROC", labelpad=10)
fig.text(
    0.075, 0.13,
    "Step line = highest observed AUROC among affordable tested models; hollow points are below the frontier.\n"
    "Observed input/output tokens (including reasoning and recorded cache use), repriced at standard OpenRouter rates.\n"
    "Excludes retries, batch/Flex discounts, and credit-purchase fees. Complete logs: reward hacking 741; sandbagging 1,767 (Astra 1,757; Opus 1,765).\n"
    "Sources: saved MALT classifications and token usage; OpenRouter model/endpoint catalogs retrieved October 2, 2026.",
    fontsize=9, color="0.35", linespacing=1.45, va="top",
)
for extension in ["png", "pdf"]:
    fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white")
plt.close(fig)
summary.to_csv(figure_data / f"{stem}.csv", index=False)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(), "price_date": price_date,
    "input_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
    "rates": rate_records,
    "cost_formula": "1000 / complete_logs * sum((prompt-read-write)*input_rate + read*cache_read_rate + write*cache_write_rate + completion*output_rate)",
    "primary_scenario": "Current standard native-endpoint prices with the observed cache-read/write pattern; no historical batch/Flex discounts.",
    "sensitivity": "nominal_no_cache_per_1000_logs_usd prices all input at uncached input rates, with neither cache-read discounts nor write surcharges.",
    "reasoning": "Already included in completion_tokens; not added again.",
    "price_frontier": "Budget step function: highest observed AUROC among individual tested models with cost at or below budget. Undefined below the cheapest model. No interpolation of model mixtures or inferred performance beyond tested models.",
    "frontier_models": {
        behavior: group.loc[group.on_price_frontier].sort_values("cost_per_1000_logs_usd").model_variant.tolist()
        for behavior, group in summary.groupby("misbehavior_type")
    },
    "limitations": "Repriced observed successful usage, not a future billing guarantee. Excludes failed/superseded attempts and missing logs; future lengths and cache hit rates may differ. Models have slightly different complete sandbagging samples.",
    "validation": "Aggregate token counts and all ten costs were cross-checked against the original per-request calculation. Cost and AUROC sample sizes match. The frozen per-request audit found no long-context pricing tiers.",
    "output_sha256": {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in sorted(list(output.glob(stem + ".*")) + list(figure_data.glob(stem + ".csv"))) if f.suffix != ".json"},
}
(output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(summary[["misbehavior_type", "model_label", "n_logs", "cost_per_1000_logs_usd", "auroc",
               "nominal_no_cache_per_1000_logs_usd"]].to_string(index=False))

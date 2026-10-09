from pathlib import Path

path = Path(__file__).resolve().parents[1]

import subprocess
import sys

# Rebuild the five figures used in the Google Doc from frozen aggregate exports.
# This entry point never downloads data, reads credentials, or calls model APIs.
scripts = [
    "make_malt_auditor_panels_tpr.py",
    "make_malt_auditor_panels_fpr.py",
    "make_malt_auditor_cost_auroc.py",
    "make_rjudge_auditor_tpr5.py",
    "make_rjudge_auditor_panels_fpr.py",
    "run_auditor_metrics.py",
    "check_draft_operating_points.py",
]
for script in scripts:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(path / "code" / script)], check=True)

# Optional record-level analyses require local, untracked research CSVs.
# python code/run_malt_ensemble_analysis.py
# python code/run_rjudge_analysis.py
# python code/check_malt_ensemble_bootstrap.py  # After run_malt_ensemble_analysis.py

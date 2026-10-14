#!/usr/bin/env python
"""Run the matrix of fits of scripts/layer2/fit.py (data vectors x second-layer models x volumes), skipping runs
that already have samples, and collect the posterior biases in outputs/layer2/bias_table.json. Example:

    python scripts/layer2/run_all.py --volumes 1 8 25 64
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "outputs/layer2/fits"
MODELS = {"gauss": [], "tree": [], "pade_l": [], "pade_l_cal": ["--fix-calibrated"],
          "pade_l_prior0.5": ["--calibrated-prior", "0.5"],
          "pade_l_cal_fromadjacent": ["--fix-calibrated", "--calibration", "adjacent"],
          "pade_l_cal_fromnonadjacent": ["--fix-calibrated", "--calibration", "nonadjacent"],
          "pade_l_cal_fromadjacent_margper_l": ["--fix-calibrated", "--calibration", "adjacent", "--marg", "per_l"],
          "pade_l_cal_fromnonadjacent_margper_l": ["--fix-calibrated", "--calibration", "nonadjacent", "--marg", "per_l"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", nargs="+", default=["baseline", "conservative", "adjacent"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--volumes", nargs="+", type=float, default=[1, 8, 25, 64])
    parser.add_argument("--cov", choices=("analytic", "mock"), default="analytic")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    table = {}
    for vector in args.vectors:
        for model in args.models:
            for volume in args.volumes:
                tag = f"{vector}_{model}{'_mock' if args.cov == 'mock' else ''}_V{volume:g}"
                if args.force or not (FITS / tag / "samples.npz").exists():
                    base = "pade_l" if model.startswith("pade_l") else model
                    cmd = [sys.executable, str(ROOT / "scripts/layer2/fit.py"), "--vector", vector, "--model", base,
                           "--volume", f"{volume:g}", "--cov", args.cov, *MODELS[model]]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    lines = [x for x in result.stdout.splitlines() if tag in x]
                    print("\n".join(lines) if lines else f"{tag}: FAILED\n{result.stderr[-2000:]}", flush=True)
                path = FITS / tag / "summary.json"
                if path.exists():
                    s = json.loads(path.read_text())
                    table[tag] = dict(chi2=s["chi2"], ndata=s["ndata"], nvaried=s["nvaried"],
                                      bias_sigma=s.get("bias_sigma"), posterior={n: s["posterior"][n] for n in
                                                                                ("omega_cdm", "logA")} if "posterior" in s else None,
                                      bestfit=s["bestfit"])
    out = ROOT / f"outputs/layer2/bias_table{'_mock' if args.cov == 'mock' else ''}.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

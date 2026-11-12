"""Participant-stratified bootstrap intervals for the immutable VA1 OOF predictions.

This is conditional on the already saved out-of-fold predictions: it resamples
participants but does not refit the model or estimate training-set variation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "baseline_oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "results" / "va1_oof_uncertainty_20261031"
REQUIRED = {"participant_folder", "true_label", "predicted_label", "predicted_pd_probability"}
METRICS = ("balanced_accuracy", "roc_auc", "sensitivity", "specificity")


def metric_values(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    positive = y == 1
    negative = y == 0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "sensitivity": float(np.mean(pred[positive] == 1)),
        "specificity": float(np.mean(pred[negative] == 0)),
    }


def analyze(frame: pd.DataFrame, *, replicates: int = 20_000, seed: int = 20261031) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["participant_folder"].isna().any() or frame["participant_folder"].duplicated().any():
        raise ValueError("Expected exactly one OOF row per non-missing participant_folder")
    for col in ("true_label", "predicted_label"):
        if not set(frame[col].dropna().unique()).issubset({0, 1}) or frame[col].isna().any():
            raise ValueError(f"{col} must contain only binary 0/1 values")
    if set(frame.true_label.unique()) != {0, 1}:
        raise ValueError("Both classes must be present")
    prob = pd.to_numeric(frame.predicted_pd_probability, errors="coerce").to_numpy(float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("predicted_pd_probability must be finite and in [0, 1]")
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    y = frame.true_label.to_numpy(int)
    pred = frame.predicted_label.to_numpy(int)
    point = metric_values(y, pred, prob)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    boot = np.empty((replicates, len(METRICS)), dtype=float)
    for i in range(replicates):
        # Fixed class counts keep all metrics defined and retain observed prevalence.
        ix = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        vals = metric_values(y[ix], pred[ix], prob[ix])
        boot[i] = [vals[name] for name in METRICS]
    rows = []
    for j, name in enumerate(METRICS):
        low, high = np.quantile(boot[:, j], [0.025, 0.975])
        rows.append({"metric": name, "estimate": point[name], "ci_method": "participant-stratified percentile bootstrap, conditional on fixed OOF predictions", "ci_level": 0.95, "ci_lower": float(low), "ci_upper": float(high), "n_participants": len(frame), "n_hc": int((y == 0).sum()), "n_pd": int((y == 1).sum()), "bootstrap_replicates": replicates, "seed": seed})
    draws = pd.DataFrame(boot, columns=METRICS)
    return pd.DataFrame(rows), draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input.read_bytes()
    frame = pd.read_csv(args.input)
    summary, draws = analyze(frame, replicates=args.replicates, seed=args.seed)
    summary.to_csv(args.output / "metric_intervals.csv", index=False)
    draws.to_csv(args.output / "bootstrap_draws.csv", index=False)
    config = {"experiment_id": "VA1-OOF-UNCERTAINTY-20261031", "input": str(args.input.relative_to(ROOT)), "input_sha256": hashlib.sha256(source).hexdigest(), "n_rows": len(frame), "n_unique_participants": int(frame.participant_folder.nunique()), "replicates": args.replicates, "seed": args.seed, "resampling_unit": "participant", "stratification": "true label; fixed class counts", "predictions_refit": False, "interpretation": "intervals are conditional on the fixed saved OOF predictions; not external validation or training-procedure uncertainty"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = ["# VA1 OOF participant uncertainty", "", "This experiment summarizes uncertainty in the existing VA1 out-of-fold predictions. It does not modify or rerun the VA1 baseline.", "", f"- Participants: {len(frame)} ({int((frame.true_label == 0).sum())} HC, {int((frame.true_label == 1).sum())} PD)", f"- Bootstrap: {args.replicates:,} class-stratified participant resamples; seed {args.seed}", "- Predictions were held fixed; no classifier was fitted.", "", "## Results", "", "| Metric | Estimate | 95% interval |", "|---|---:|---:|"]
    for _, row in summary.iterrows():
        report.append(f"| {row.metric} | {row.estimate:.3f} | [{row.ci_lower:.3f}, {row.ci_upper:.3f}] |")
    report += ["", "## Interpretation and limits", "", "The intervals describe participant-sampling variability conditional on the saved cross-validated predictions. Participants' predictions share fitted fold models, so this bootstrap does not capture model refitting or fold-assignment variability. It is not an external validation interval and does not support diagnostic claims. The small cohort and internal cross-validation remain substantial limits.", "", "## Reproduction", "", "Run `.venv/bin/python src/va1_oof_uncertainty.py --output results/va1_oof_uncertainty_reproduction` from the project root to reproduce in a fresh directory. Inputs are read-only; the runner refuses to overwrite a non-empty output directory.", ""]
    (args.output / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()

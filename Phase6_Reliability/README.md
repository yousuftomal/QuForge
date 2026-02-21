# Phase 6 Reliability

This phase runs reliability-focused ablations and stress tests for publishable evidence.

## What it does

- Builds a high-trust measurement subset from `Dataset/measurement_dataset_public_bootstrap.csv`.
- Runs Phase 4 ablations and validation in isolated artifact folders.
- Computes risk-coverage curves and abstention gains from each trained bundle.
- Selects primary and secondary Phase 4 variants by a reliability rank.
- Runs Phase 5 stress tests (strict and gate-relaxed settings) using the chosen Phase 4 bundles.
- Writes paper-ready outputs:
  - `phase6_phase4_variant_summary.csv`
  - `phase6_risk_coverage_curves.csv`
  - `phase6_phase5_stress_summary.csv` (unless `--skip-phase5`)
  - `phase6_report.md`
  - `phase6_summary.json`

## Run

```powershell
Dataset\.venv310\Scripts\python Phase6_Reliability\run_phase6_reliability.py
```

Optional quick run without Phase 5:

```powershell
Dataset\.venv310\Scripts\python Phase6_Reliability\run_phase6_reliability.py --skip-phase5
```

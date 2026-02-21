# Phase 4 Coherence Prediction

Phase 4 trains a probabilistic coherence predictor with uncertainty and OOD flags.

What it provides:
- Probabilistic predictions for `t1_us` and `t2_us` (p10/p50/p90)
- Interval calibration and OOD validation (ID vs OOD)
- Geometry/physics feature risk ranking for low-coherence regions
- High-uncertainty design shortlist for active-learning measurements

Important current-data note:
- `t1_estimate_us` in this dataset is numerically stable and used directly.
- `t2_estimate_us` has extreme dynamic range in generated proxy data, so Phase 4 uses robust clipping + log-target modeling.
- Validation therefore reports both standard MAE and `log10_MAE` (especially relevant for `t2_us`).

## Real measured data workflow

1. Fill raw lab file from template:
- `Dataset/measurement_raw.template.csv`

2. Build standardized measurement dataset + QA report:

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\build_measurement_dataset.py --raw-csv Dataset\measurement_raw.csv --output-csv Dataset\measurement_dataset.csv --report-json Dataset\measurement_dataset.report.json
```

3. Retrain in hybrid mode:

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset.csv
Dataset\.venv310\Scripts\python Phase4_Coherence\validate_phase4_coherence.py --measurement-csv Dataset\measurement_dataset.csv
```

Detailed instructions: `Phase4_Coherence/MEASUREMENT_DATA_GUIDE.md`.

## Install

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase4_Coherence\requirements-phase4.txt
```

## Train (default proxy/hybrid-auto)

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py
```

## Validate

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\validate_phase4_coherence.py
```

## Predict (from geometry + optional Phase1 fill)

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\predict_coherence_phase4.py --pad-width-um 500 --pad-height-um 500 --gap-um 30 --junction-area-um2 0.03
```

## Predict (from Phase 3 candidate)

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\predict_coherence_phase4.py --candidate-json Phase3_InverseDesign\artifacts\inverse_design_example.json --rank 1 --output-json Phase4_Coherence\artifacts\coherence_prediction_example.json
```

## Artifacts

- `Phase4_Coherence/artifacts/phase4_coherence_bundle.joblib`
- `Phase4_Coherence/artifacts/phase4_training_summary.json`
- `Phase4_Coherence/artifacts/phase4_validation_report.json`
- `Phase4_Coherence/artifacts/phase4_feature_risk_report.csv`
- `Phase4_Coherence/artifacts/phase4_high_uncertainty_candidates.csv`
- `Phase4_Coherence/artifacts/coherence_prediction_example.json`
- `Dataset/measurement_dataset.csv` (built from real lab data)
- `Dataset/measurement_dataset.report.json`

## Bootstrap from public datasets (Phase 4.5)

If lab-only measured rows are unavailable, you can bootstrap with public hardware data via `Phase4_5_PublicData/`.

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\fetch_public_datasets.py
Dataset\.venv310\Scripts\python Phase4_5_PublicData\build_public_canonical_dataset.py
Dataset\.venv310\Scripts\python Phase4_5_PublicData\map_public_to_internal.py --max-distance 1.2 --min-confidence 0.55
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset_public_bootstrap.csv --measured-weight 3.0 --proxy-weight 1.0
```

`train_phase4_coherence.py` now reads optional `confidence_weight` from measurement CSV and uses it as sample weighting.

## Phase 4.6 trace-fit augmentation (optional, low-cost)

Use this when you want to bootstrap extra measured coherence rows from public raw traces:

```powershell
Dataset\.venv310\Scripts\python Phase4_6_TraceCoherence\augment_with_zenodo_8004359_traces.py
Dataset\.venv310\Scripts\python Phase4_5_PublicData\map_public_to_internal.py --public-csv Dataset\public_sources\silver\public_measurements_canonical_augmented.csv --max-distance 1.4 --min-confidence 0.40
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset_public_bootstrap.csv --measured-weight 1.0 --proxy-weight 1.0
```

This path remains bootstrap-grade and uses conservative weighting.

# Measurement Data Guide

This project cannot generate real measured coherence by simulation.
`Dataset/measurement_dataset.csv` must come from lab hardware measurements.

## What "real measured data" means here

For each fabricated/measured qubit design, you need:
- a link to dataset identity (`row_index` preferred, `design_id` accepted)
- measured coherence values from experiment (`measured_t1_us`, `measured_t2_us`)
- optional measured spectroscopy (`measured_freq_01_GHz`, `measured_anharmonicity_GHz`)
- optional metadata (`chip_id`, `cooldown_id`, `measurement_date_utc`, `notes`)

These values should come from your readout pipeline (lab notebooks, instrument exports, analysis scripts), not from synthetic model output.

## Recommended workflow

1. Fill raw template from lab records:
- Start from `Dataset/measurement_raw.template.csv`.
- Keep one row per measured device/cooldown result.
- Use real measured numbers in microseconds for T1/T2.

2. Build + QA standardized dataset:

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\build_measurement_dataset.py --raw-csv Dataset\measurement_raw.csv --output-csv Dataset\measurement_dataset.csv --report-json Dataset\measurement_dataset.report.json
```

3. Check QA report:
- Open `Dataset/measurement_dataset.report.json`.
- Verify `rows_final`, `dropped_unmapped_rows`, and `id_mismatch_rows` are acceptable.

4. Train Phase 4 with measured data:

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset.csv
Dataset\.venv310\Scripts\python Phase4_Coherence\validate_phase4_coherence.py --measurement-csv Dataset\measurement_dataset.csv
```

## ID mapping guidance

- Best: provide `row_index` directly from `Dataset/final_dataset_single.csv`.
- Also supported: `design_id` if unique (it is unique in current dataset).
- If both are provided, they must match the source dataset mapping.

## Minimum amount to start

- Usable first milestone: 30+ measured rows.
- Better reliability: 50-100 measured rows.
- Stronger competition-grade model: 200+ measured rows with diverse chips/cooldowns.

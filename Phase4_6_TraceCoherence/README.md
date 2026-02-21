# Phase 4.6 Trace Coherence Augmentation

Extracts T1/T2 from Zenodo 8004359 raw decay traces and appends them to the public canonical dataset.

Workflow:

1. Build augmented canonical CSV from trace fits:

```powershell
Dataset\.venv310\Scripts\python Phase4_6_TraceCoherence\augment_with_zenodo_8004359_traces.py
```

2. Map augmented public rows to internal measurement schema (conservative):

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\map_public_to_internal.py --public-csv Dataset\public_sources\silver\public_measurements_canonical_augmented.csv --max-distance 1.6 --min-confidence 0.30
```

3. Train with low measured weight (safe for bootstrap labels):

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset_public_bootstrap.csv --measured-weight 1.0 --proxy-weight 1.0
Dataset\.venv310\Scripts\python Phase4_Coherence\validate_phase4_coherence.py --measurement-csv Dataset\measurement_dataset_public_bootstrap.csv
```

Notes:
- Frequency is inferred only when defensible (Device B + Fig5c spec map).
- Rows without inferred frequency remain in canonical but are not mapped.
- Mapping now multiplies distance confidence by optional `source_confidence`.

4. One-command run:

```powershell
Dataset\.venv310\Scripts\python Phase4_6_TraceCoherence\run_phase46_pipeline.py
```

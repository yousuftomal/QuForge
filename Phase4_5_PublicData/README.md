# Phase 4.5 Public Data Pipeline

This phase ingests public superconducting-qubit datasets into a canonical measurement table and maps conservative matches into the internal `measurement_dataset` schema.

## Source Coverage

The fetch stage now expands and downloads from:

- `SQuADDS/SQuADDS_DB` (measured database + qubit parquet)
- Zenodo records: `4336924`, `8004359`, `18045662`, `15364358`, `13808824`, `14628434`
- Data.gov/NIST packages: `mds2-2516`, `mds2-2932`, `mds2-3027`, `mds2-2756`

## Files

- `fetch_public_datasets.py`: resolves record/package APIs and downloads source files into bronze storage
- `build_public_canonical_dataset.py`: parses CSV/TSV/TXT/JSON/ZIP/XLSX/Parquet into canonical silver CSV
- `map_public_to_internal.py`: nearest-neighbor mapping from public rows to internal rows with confidence scores

## Workflow

1. Install dependencies:

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase4_5_PublicData\requirements-phase45.txt
```

2. Fetch all relevant public sources (large groups enabled, per-file size capped):

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\fetch_public_datasets.py --include-large --max-download-mb 350
```

3. Build canonical public dataset:

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\build_public_canonical_dataset.py
```

4. Conservative mapping into internal measurement schema:

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\map_public_to_internal.py --max-distance 1.6 --min-confidence 0.30
```

5. Train Phase 4 in hybrid mode with confidence-aware weighting:

```powershell
Dataset\.venv310\Scripts\python Phase4_Coherence\train_phase4_coherence.py --label-mode hybrid --measurement-csv Dataset\measurement_dataset_public_bootstrap.csv --measured-weight 1.0 --proxy-weight 1.0
```

## Notes

- `--include-large` enables source groups tagged large; `--max-download-mb` still blocks oversized individual files.
- All fetch/build/map steps emit JSON reports for auditability.
- Public bootstrap rows are still proxy labels and must be treated with lower trust than lab-measured rows.

### Optional: Large Raw-Source Trace Augmentation

After canonical build, run the large-file augmentation step to attempt conservative T1 extraction from very large raw files (for example `mds2-2516 fig_4a_*` and `zenodo_15364358 figS4d`).

```powershell
Dataset\.venv310\Scripts\python Phase4_5_PublicData\augment_large_raw_sources.py --input-csv Dataset\public_sources\silver\public_measurements_canonical.csv --output-csv Dataset\public_sources\silver\public_measurements_canonical.csv
```

Notes:
- This step is conservative and keeps only physically plausible fitted rows.
- It also audits large HDF5 files and reports whether direct T1/T2 datasets exist.
- Rows from this step are usually missing `freq_01`, so they may not map into internal bootstrap without additional mapping logic.

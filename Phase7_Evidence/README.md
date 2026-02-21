# Phase 7 Evidence

Phase 7 hardens claims for paper and lab deployment.

## Deliverables

- Multi-seed reliability stability analysis (via repeated Phase 6 runs with `--skip-phase5`)
- Source holdout validation (leave-one-source-out on mapped measurement sources)
- Paper-ready summary artifacts (`csv`, `json`, `md`)
- Non-CLI web interface for running closed-loop design batches

## Run Phase 7

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\run_phase7_evidence.py
```

Optional (custom seeds):

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\run_phase7_evidence.py --seeds 42,123,777,2026
```

## Launch Web Interface (non-CLI usage once opened)

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase7_Evidence\requirements-phase7.txt
$env:PYTHONUTF8='1'
Dataset\.venv310\Scripts\python -m streamlit run Phase7_Evidence\webapp\app.py
```

## Key output files

- `Phase7_Evidence/artifacts/phase7_multiseed_raw.csv`
- `Phase7_Evidence/artifacts/phase7_multiseed_summary.csv`
- `Phase7_Evidence/artifacts/source_holdout/phase7_source_holdout_summary.csv`
- `Phase7_Evidence/artifacts/paper/phase7_paper_claims_table.csv`
- `Phase7_Evidence/artifacts/phase7_report.md`
- `Phase7_Evidence/artifacts/phase7_summary.json`

## Publication Figure Pack

Generate paper-ready figures from the large sweep:

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\generate_publication_figures.py --input-dir Phase7_Evidence\artifacts_large_sweep --output-dir Phase7_Evidence\artifacts_large_sweep\figures
```

Outputs:
- `figure1_rank_stability.png`
- `figure2_ood_boxplots.png`
- `figure3_source_holdout_table.png`
- `figure_captions.md`
- `figure_manifest.json`

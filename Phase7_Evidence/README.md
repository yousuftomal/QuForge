# Phase 7 Evidence

Phase 7 hardens evidence for research validation and lab evaluation.

## Deliverables

- Multi-seed reliability stability analysis (via repeated Phase 6 runs with `--skip-phase5`)
- Source holdout validation (leave-one-source-out on mapped measurement sources)
- Summary artifacts (`csv`, `json`, `md`)

## Run Phase 7

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\run_phase7_evidence.py
```

Optional (custom seeds):

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\run_phase7_evidence.py --seeds 42,123,777,2026
```

## Private UI Note

An internal non-CLI web interface exists for interactive closed-loop runs, but its code is intentionally excluded from this public repository.

## Key output files

- `Phase7_Evidence/artifacts/phase7_multiseed_raw.csv`
- `Phase7_Evidence/artifacts/phase7_multiseed_summary.csv`
- `Phase7_Evidence/artifacts/source_holdout/phase7_source_holdout_summary.csv`
- `Phase7_Evidence/artifacts/phase7_report.md`
- `Phase7_Evidence/artifacts/phase7_summary.json`

## Figure Pack

Generate figures from the large sweep:

```powershell
Dataset\.venv310\Scripts\python Phase7_Evidence\generate_publication_figures.py --input-dir Phase7_Evidence\artifacts_large_sweep --output-dir Phase7_Evidence\artifacts_large_sweep\figures
```

Outputs:
- `figure1_rank_stability.png`
- `figure2_ood_boxplots.png`
- `figure3_source_holdout_table.png`
- `figure_captions.md`
- `figure_manifest.json`

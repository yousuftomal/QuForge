# Phase 5 Closed-Loop Candidate Engine

This phase executes an automated loop:

1. Build target physics set (random by default, or from CSV).
2. Run Phase 3 inverse design per target.
3. Score each candidate with Phase 4 coherence uncertainty + OOD checks.
4. Apply hard acceptance gates.
5. Output ranked candidate batch and selected shortlist.

## Run

```powershell
Dataset\.venv310\Scripts\python Phase5_ClosedLoop\run_phase5_closed_loop.py
```

## Optional target CSV

CSV columns:
- `target_id` (optional)
- `freq_01_GHz` (required)
- `anharmonicity_GHz` (required)
- `EJ_GHz` (optional)
- `EC_GHz` (optional)
- `charge_sensitivity_GHz` (optional)

```powershell
Dataset\.venv310\Scripts\python Phase5_ClosedLoop\run_phase5_closed_loop.py --targets-csv Phase5_ClosedLoop\targets.template.csv
```

## Outputs

- `Phase5_ClosedLoop/artifacts/phase5_targets_used.csv`
- `Phase5_ClosedLoop/artifacts/phase5_candidate_batch.csv`
- `Phase5_ClosedLoop/artifacts/phase5_selected_candidates.csv`
- `Phase5_ClosedLoop/artifacts/phase5_summary.json`
- `Phase5_ClosedLoop/artifacts/phase5_report.md`
## Phase 5.1 Fabrication Handoff

Export a fabrication-ready package from the selected Phase 5 candidates:

```powershell
Dataset\.venv310\Scripts\python Phase5_ClosedLoop\export_fab_handoff.py
```

Optional controls:
- `--top-n 20`
- `--allow-ood`
- `--allow-low-confidence`

Handoff outputs:
- `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_candidates.csv`
- `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_package.json`
- `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_report.md`

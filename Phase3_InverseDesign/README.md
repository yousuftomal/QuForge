# Phase 3 Inverse Design

Phase 3 turns target physics into geometry proposals using:

- Phase 2 NN embedding (physics -> shared latent space)
- Phase 1 surrogate model (geometry -> predicted physics)
- Evolutionary search in continuous geometry space

Default mode is tuned for inverse-accuracy (best benchmarked MAE).
Robustness penalties are available via CLI flags when you want fabrication-margin-aware search.

## Install

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase3_InverseDesign\requirements-phase3.txt
```

## Run Inverse Design (accuracy mode, default)

```powershell
Dataset\.venv310\Scripts\python Phase3_InverseDesign\inverse_design_phase3.py --freq-01-ghz 5.0 --anharmonicity-ghz -0.2 --top-n 10 --output-json Phase3_InverseDesign\artifacts\inverse_design_example.json
```

## Run Inverse Design (robust mode example)

```powershell
Dataset\.venv310\Scripts\python Phase3_InverseDesign\inverse_design_phase3.py --freq-01-ghz 5.0 --anharmonicity-ghz -0.2 --top-n 10 --embedding-weight 0.05 --robustness-weight 0.2 --robust-samples 5 --fabrication-tolerance 0.05
```

## Validate

```powershell
Dataset\.venv310\Scripts\python Phase3_InverseDesign\validate_phase3.py --max-queries-per-split 40
```

## Key Artifacts

- `Phase3_InverseDesign/artifacts/inverse_design_example.json`
- `Phase3_InverseDesign/artifacts/phase3_validation_report.json`
- `Phase3_InverseDesign/artifacts/phase3_validation_report.md`

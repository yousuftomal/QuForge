# Phase 2 Embedding

This folder now contains two Phase 2 pipelines:

- Baseline CCA/RFF pipeline (legacy)
- **Neural twin-encoder pipeline (recommended)**

Use the NN pipeline for current work.

## Install

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase2_Embedding\requirements-embedding.txt
```

## Train (NN, recommended)

```powershell
Dataset\.venv310\Scripts\python Phase2_Embedding\train_phase2_embedding_nn.py
```

## Validate (NN, recommended)

```powershell
Dataset\.venv310\Scripts\python Phase2_Embedding\validate_phase2_embedding_nn.py
```

## Inverse-design query (NN)

```powershell
Dataset\.venv310\Scripts\python Phase2_Embedding\inverse_design_nn.py --freq-01-ghz 5.0 --anharmonicity-ghz -0.2 --top-n 10
```

## Key NN artifacts

- `Phase2_Embedding/artifacts/phase2_nn_bundle.pt`
- `Phase2_Embedding/artifacts/phase2_nn_training_summary.json`
- `Phase2_Embedding/artifacts/phase2_nn_validation_report.json`
- `Phase2_Embedding/artifacts/inverse_design_nn_example.json`

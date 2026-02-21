# Phase 1 Surrogate Models

This folder contains the next-step training pipeline after dataset generation.

## What it trains

- `single` surrogate:
  - Inputs: `pad_width_um`, `pad_height_um`, `gap_um`, `junction_area_um2`
  - Outputs: `freq_01_GHz`, `anharmonicity_GHz`, `EJ_GHz`, `EC_GHz`, `charge_sensitivity_GHz`

- `coupled` surrogate:
  - Inputs: `freq_q1_GHz`, `freq_q2_GHz`, `resonator_freq_GHz`, `coupling_strength_g_GHz`
  - Outputs: `dressed_freq_q1_GHz`, `dressed_freq_q2_GHz`, `dispersive_shift_chi_GHz`

Both models use multi-output random forests and save metrics/artifacts for reuse.

## Install

From project root:

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Phase1_Surrogate\requirements-training.txt
```

## Train

From project root:

```powershell
Dataset\.venv310\Scripts\python Phase1_Surrogate\train_surrogate.py
```

Artifacts are written to `Phase1_Surrogate\artifacts\`.

## Predict

Single-qubit example:

```powershell
Dataset\.venv310\Scripts\python Phase1_Surrogate\predict_surrogate.py --mode single --pad-width-um 500 --pad-height-um 500 --gap-um 30 --junction-area-um2 0.03
```

Coupled example:

```powershell
Dataset\.venv310\Scripts\python Phase1_Surrogate\predict_surrogate.py --mode coupled --freq-q1-ghz 5.1 --freq-q2-ghz 5.3 --resonator-freq-ghz 6.8 --coupling-strength-g-ghz 0.1
```


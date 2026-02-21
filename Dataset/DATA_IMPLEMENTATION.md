# DATA.md Verification and Implementation Notes

This repository now includes `generate_data.py`, which implements the full workflow in `DATA.md`:

1. Geometry generation (Qiskit Metal)
2. EM extraction (Palace via Docker)
3. EM -> transmon parameters (EC, EJ)
4. Quantum properties (scqubits)
5. Coupled-system dataset generation
6. CSV export for downstream surrogate training

## What Was Verified and Corrected

- `EJ` conversion is now explicit and unit-correct:
  - `EJ/h (GHz) = Ic / (4*pi*e) / 1e9`
- `EC` conversion uses SI constants with Farads input:
  - `EC/h (GHz) = e^2 / (2*C*h) / 1e9`
- The rough frequency filter was corrected to transmon approximation:
  - `f01 ~ sqrt(8*EJ*EC) - EC`
- Palace output parsing is robust to filename and format variants:
  - supports `terminal-C.csv`, `capacitance_matrix.txt`, and discovered capacitance files
- Coherence outputs are normalized to microseconds with safe handling for version differences.
- scqubits interaction setup for coupled systems uses a compatibility-friendly `add_interaction` callable form.

## Run Instructions

Install Python deps:

```bash
pip install -r requirements-dataset.txt
```

Ensure Docker is installed and Palace image is available:

```bash
docker pull ghcr.io/awslabs/palace:latest
```

Dry run (recommended first):

```bash
python generate_data.py --dry-run --skip-palace --skip-geometry
```

Full pipeline with Qiskit Metal + Palace:

```bash
python generate_data.py
```

Outputs:

- `final_dataset_single.csv`
- `final_dataset_coupled.csv`

## Practical Notes

- Palace boundary/material attributes depend on your mesh tags and may need project-specific adjustment.
- Use `--max-designs N` and `--max-pairs M` to control compute while validating setup.
- `--skip-palace` provides an analytic capacitance fallback for smoke tests.

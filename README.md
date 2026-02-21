# QuForge

AI-assisted superconducting qubit design pipeline with staged phases from physics-driven data generation to reliability-aware candidate selection.

## Repository Structure

- `Dataset/`: data generation scripts, templates, and dataset commands
- `Phase1_Surrogate/`: surrogate models for fast property prediction
- `Phase2_Embedding/`: embedding + inverse retrieval models
- `Phase3_InverseDesign/`: constrained inverse-design search engine
- `Phase4_Coherence/`: coherence prediction and validation
- `Phase4_5_PublicData/`: public data ingestion/mapping utilities
- `Phase5_ClosedLoop/`: closed-loop candidate engine and fab handoff export
- `Phase6_Reliability/`: reliability ablations and stress evaluation
- `Phase7_Evidence/`: evidence hardening, figures, and web interface
- `run_command.txt`: consolidated command reference

## Quick Start

Use the project environment and run commands from `run_command.txt`:

```powershell
Dataset\.venv310\Scripts\python -m pip install -r Dataset\requirements-dataset.txt
Dataset\.venv310\Scripts\python Dataset\generate_data.py --dry-run --workdir Dataset
```

For full strict generation and paper rerun, use the "Final paper rerun" block in `run_command.txt`.

## Notes

- Generated artifacts, virtual environments, and large downloaded sources are intentionally git-ignored.
- License: Apache 2.0 (`LICENSE`).

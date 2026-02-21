# Project Analysis (Current State)

Updated: 2026-02-21

## 1) Objective and Scope

This project implements an end-to-end AI pipeline for superconducting qubit design:

- Phase 1: fast surrogate models
- Phase 2: geometry/physics embedding + inverse retrieval
- Phase 3: inverse-design optimizer
- Phase 4: coherence prediction with uncertainty/OOD checks
- Phase 4.5/4.6: public-measurement ingestion and conservative bootstrap mapping
- Phase 5: closed-loop candidate generation + fabrication handoff export

Primary target: robust design ranking and shortlisting under uncertainty, with eventual calibration to real measured lab data.

## 2) Final Data Inventory

Current core datasets:

- `Dataset/final_dataset_single.csv`: 10,000 rows
- `Dataset/final_dataset_coupled.csv`: 80,000 rows
- `Dataset/public_sources/silver/public_measurements_canonical.csv`: 2,224 rows
- `Dataset/public_sources/silver/public_measurements_canonical_augmented.csv`: 2,253 rows
- `Dataset/measurement_dataset_public_bootstrap.csv`: 179 mapped rows

Public fetch status (`Dataset/public_sources/fetch_report.json`):

- candidates: 93
- downloaded: 1
- cached: 92
- skipped: 0
- errors: 0

## 3) Phase-by-Phase Results

### Phase 1: Surrogate Models

Artifacts:
- `Phase1_Surrogate/artifacts/single_metadata.json`
- `Phase1_Surrogate/artifacts/coupled_metadata.json`

Key results:
- Single-qubit `freq_01_GHz`: R2 = 0.9986579463, MAE = 0.029334268 GHz
- Coupled `dispersive_shift_chi_GHz`: R2 = 0.9999680597

Finding:
- Surrogates are strong and production-usable for fast screening.

### Phase 2: Embedding + Retrieval

Artifact:
- `Phase2_Embedding/artifacts/phase2_nn_validation_report.json`

Key results:
- Retrieval top-5 accuracy: ID = 0.8614, OOD = 0.4397
- Inverse reranked frequency MAE: ID = 0.004372 GHz, OOD = 0.002872 GHz

Finding:
- Embedding/retrieval is good on ID and reasonable on OOD; reranking materially improves inverse queries.

### Phase 3: Inverse Design

Artifact:
- `Phase3_InverseDesign/artifacts/phase3_validation_report.json`

Key results:
- Mean freq MAE (ID): 0.002568 GHz
- Mean freq MAE (OOD): 0.001959 GHz

Finding:
- Optimizer consistently beats baseline nearest retrieval and is stable.

### Phase 4: Coherence Model

Artifacts:
- `Phase4_Coherence/artifacts/phase4_training_summary.json`
- `Phase4_Coherence/artifacts/phase4_validation_report.json`

Current training context:
- `measurement_rows_matched`: 179 (from public bootstrap mapping)

Current validation metrics:
- ID t1 MAE: 0.000670 us
- OOD t1 MAE: 0.001413 us
- ID t2 log10_MAE: 0.409842
- OOD t2 log10_MAE: 0.454910

Finding:
- Pipeline is functional and calibrated for ranking with uncertainty, but coherence prediction remains limited by scarcity of true geometry-linked measured data.

### Phase 4.5/4.6: Public Data Ingestion and Mapping

Artifacts:
- `Dataset/public_sources/silver/public_measurements_canonical.report.json`
- `Dataset/public_sources/silver/public_measurements_canonical_augmented.report.json`
- `Dataset/measurement_dataset_public_bootstrap.report.json`
- `Dataset/public_sources/silver/public_measurements_large_raw_augmented.report.json`

Canonical (before 4.6 augment):
- rows_total: 2,223
- rows_with_t1: 1,223
- rows_with_t2: 2,223
- rows_with_freq: 1,005

After 4.6 trace augmentation:
- rows_output: 2,253
- rows_output_with_freq: 1,011

Public -> internal mapping:
- public_rows_input: 2,253
- mapped_rows_final: 179
- rejected_missing_freq: 1,242
- source_counts: DataGov mds2-3027 (174), SQuADDS (4), Zenodo 8004359 tracefit (1)

Finding:
- Public-data scale increased substantially, but only a subset maps conservatively to internal design rows.

### Phase 5: Closed Loop + Handoff

Artifacts:
- `Phase5_ClosedLoop/artifacts/phase5_summary.json`
- `Phase5_ClosedLoop/artifacts/phase5_selected_candidates.csv`
- `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_package.json`

Latest rerun (using latest Phase 4 model):
- targets_total: 24
- candidates_total: 192
- candidates_pass_all: 33
- selected_total: 30
- selected_ood_rate: 0.0

Handoff export:
- selected_input: 30
- eligible_after_gates: 30
- exported: 20
- fallback used: false

Finding:
- Closed-loop engine and fabrication handoff are operational and reproducible.

## 4) Public Sources Covered

Integrated source families:

- SQuADDS (Hugging Face)
- Zenodo records: 4336924, 8004359, 18045662, 15364358, 13808824, 14628434
- Data.gov/NIST packages: mds2-2516, mds2-2932, mds2-3027, mds2-2756

All currently configured candidates now download without skip/error under high size cap.

## 5) Large-File Investigation Outcome

Requested concern: potential value lost due to size limits.

Actions taken:
- Re-fetched with `--max-download-mb 2000`.
- Added `Phase4_5_PublicData/augment_large_raw_sources.py`.
- Parsed large mds2-2516 raw matrices (`fig_4a_pex/tlist`) and zenodo 15364358 figure-fit CSV.
- Inspected 706 MB HDF5 structure in zenodo 15364358.

Outcome:
- Large files were successfully acquired and evaluated.
- Conservative fitter retained only 1 physically plausible additional row (from zenodo figS4d).
- Large mds2-2516 matrix traces did not produce reliable, physically plausible T1 under strict bounds.
- HDF5 contains rich raw decoder/measurement tensors but no direct T1/T2 arrays; it requires a custom decoding/physics extraction pipeline beyond current mapping assumptions.

## 6) What Is Solved vs Not Solved

Solved:
- End-to-end pipeline from dataset -> surrogate -> embedding -> inverse design -> coherence scoring -> closed-loop shortlist -> fab handoff.
- Public data ingestion and conservative bootstrap mapping at scale.
- Reproducible run commands and generated artifacts.

Not yet solved (main gap):
- High-confidence absolute coherence prediction tied to real geometry + real measured outcomes at substantial scale.
- True internal measured dataset (`Dataset/measurement_dataset.csv`) is still missing; bootstrap is mostly proxy/freq-only mapping.

## 7) Practical Readiness Assessment

Current system status:
- Strong research prototype and candidate-ranking engine.
- Good for narrowing design space and reducing clearly weak candidates.
- Not yet equivalent to a lab-calibrated production predictor for absolute T1/T2 targets.

## 8) Latest Optional Next Actions

If continuing immediately:
1. Keep Phase 5 as-is and start fabrication shortlist review from `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_candidates.csv`.
2. Build a strict high-trust subset (e.g., higher-confidence + richer label rows) and compare Phase 4 metrics against current mixed bootstrap.
3. Add real measured internal data (`measurement_raw.csv` -> `measurement_dataset.csv`) to close the core accuracy gap.

## 9) Key Artifact Paths

- `Dataset/public_sources/fetch_report.json`
- `Dataset/public_sources/silver/public_measurements_canonical.report.json`
- `Dataset/public_sources/silver/public_measurements_large_raw_augmented.report.json`
- `Dataset/public_sources/silver/public_measurements_canonical_augmented.report.json`
- `Dataset/measurement_dataset_public_bootstrap.report.json`
- `Phase1_Surrogate/artifacts/single_metadata.json`
- `Phase1_Surrogate/artifacts/coupled_metadata.json`
- `Phase2_Embedding/artifacts/phase2_nn_validation_report.json`
- `Phase3_InverseDesign/artifacts/phase3_validation_report.json`
- `Phase4_Coherence/artifacts/phase4_training_summary.json`
- `Phase4_Coherence/artifacts/phase4_validation_report.json`
- `Phase5_ClosedLoop/artifacts/phase5_summary.json`
- `Phase5_ClosedLoop/artifacts/handoff/fab_handoff_package.json`

## 10) Phase 7 (Evidence Hardening) Results (2026-02-21)

Artifacts:
- `Phase7_Evidence/artifacts/phase7_multiseed_raw.csv`
- `Phase7_Evidence/artifacts/phase7_multiseed_summary.csv`
- `Phase7_Evidence/artifacts/source_holdout/phase7_source_holdout_summary.csv`
- `Phase7_Evidence/artifacts/paper/phase7_paper_claims_table.csv`
- `Phase7_Evidence/artifacts/phase7_report.md`
- `Phase7_Evidence/artifacts/phase7_summary.json`

What was executed:
- Multi-seed Phase 6 reliability sweeps (seeds: 42, 123, 777; Phase 5 skipped for speed).
- Source holdout retrains/evaluation for mapped sources with enough rows (`DataGov:NIST:mds2-3027`, `SQuADDS`).
- Paper claims table generation.
- Non-CLI web interface implementation via Streamlit (`Phase7_Evidence/webapp/app.py`).

Key findings:
- `proxy_baseline` remained top variant across all tested seeds (`top1_count=3`, rank mean=1.0).
- Ranking order was stable across seeds (rank std=0 for all variants under current setup).
- Out-of-source holdout errors are high, confirming limited cross-source absolute-coherence transfer with current mapped labels.

Practical interpretation:
- Phase 7 strengthens reproducibility and reporting quality.
- The main bottleneck remains real, geometry-linked measured coherence at scale.
- Web interface is now available for non-CLI workflow while preserving Phase 5/6 logic.

## 11) Phase 7 Large-Sweep Validation (10 Seeds, 2026-02-21)

Run:
- `Phase7_Evidence/run_phase7_evidence.py --seeds 42,123,777,2026,31415,27182,16180,999,1337,4242 --output-dir Phase7_Evidence/artifacts_large_sweep`

Checks performed:
- Verified all 10 seeds completed and produced 5-variant outputs each (50 rows total).
- Verified top-1 variant frequency and rank variance.
- Verified holdout summaries regenerated in the large-sweep output folder.

Findings:
- Top-1 by seed: `proxy_baseline` won 9/10 seeds; `hybrid_high_trust` won 1/10 (seed 27182).
- Mean rank across seeds:
  - `proxy_baseline`: 1.10 (std 0.316)
  - `hybrid_full_conf`: 2.50 (std 0.527)
  - `hybrid_high_trust`: 2.80 (std 1.135)
  - `hybrid_no_embedding_ood`: 3.60 (std 0.516)
  - `hybrid_no_conf_weight`: 5.00 (std 0.000)
- Proxy baseline variability (across 10 seeds):
  - OOD t1 MAE mean 0.000423, std 0.000038 (CV ~9.1%)
  - OOD t2 log10 MAE mean 0.398510, std 0.014903 (CV ~3.7%)
- Source holdout remains weak (same conclusion as before): high out-of-source errors, indicating limited absolute transfer under current mapped labels.

Conclusion:
- Larger sweep confirms the earlier decision is mostly stable: `proxy_baseline` remains the most reliable default under current data regime, but rank is not perfectly deterministic across seeds.

## 12) Publication Figure Pack Generated (Large Sweep)

Script:
- `Phase7_Evidence/generate_publication_figures.py`

Command run:
- `Dataset/.venv310/Scripts/python Phase7_Evidence/generate_publication_figures.py --input-dir Phase7_Evidence/artifacts_large_sweep --output-dir Phase7_Evidence/artifacts_large_sweep/figures`

Outputs:
- `Phase7_Evidence/artifacts_large_sweep/figures/figure1_rank_stability.png`
- `Phase7_Evidence/artifacts_large_sweep/figures/figure2_ood_boxplots.png`
- `Phase7_Evidence/artifacts_large_sweep/figures/figure3_source_holdout_table.png`
- `Phase7_Evidence/artifacts_large_sweep/figures/figure_captions.md`
- `Phase7_Evidence/artifacts_large_sweep/figures/figure_manifest.json`

Validation:
- All figures generated with non-trivial resolution and file sizes.
- Manifest paths resolve correctly.
- Captions file is present for paper text integration.


## 13) Strict No-Fit Rebuild + Source-Aware Calibration (2026-02-21)

Code-level changes implemented:
- `Dataset/generate_data.py`: total `T2` now computed from `T1` and `Tphi` (`1/T2 = 1/(2T1) + 1/Tphi`) instead of using pure dephasing directly as `T2`.
- `Phase4_5_PublicData/build_public_canonical_dataset.py`: fitted/model curve columns are excluded by default during canonical extraction.
- `Phase4_5_PublicData/map_public_to_internal.py`: fitted/model/tracefit rows are excluded by default (`--include-fitted-curves` opt-in).
- `Phase4_Coherence/train_phase4_coherence.py`: added source-aware calibration layer (per-source log-domain canonicalization of measured labels before hybrid target construction).
- `Phase4_Coherence/validate_phase4_coherence.py`, `Phase6_Reliability/run_phase6_reliability.py`, `Phase7_Evidence/run_phase7_evidence.py`: aligned evaluation/holdout logic with source-aware canonicalization.
- `Phase5_ClosedLoop/run_phase5_closed_loop.py`, `Phase4_Coherence/predict_coherence_phase4.py`: added source-profile aware projection/inflation hooks.
- `Phase7_Evidence/run_phase7_evidence.py`: paper report now explicitly states validated claim scope.

Rebuilt data and rerun outputs:
- Canonical rebuild: `rows_total=1244`, `rows_with_freq=26` (non-fitted default).
- Augmented canonical: `rows_output=1273`, `rows_output_with_freq=32`.
- Public->internal mapping (strict no-fit default):
  - `public_rows_input=5`
  - `mapped_rows_final=4`
  - `rejected_fitted_curve_rows=1268`
  - mapped source: SQuADDS only (4 rows, `freq+anh` matches)
- Phase 4 rerun (`hybrid`, strict bootstrap):
  - ID t1 MAE: `0.000548 us`
  - OOD t1 MAE: `0.001187 us`
  - ID t2 log10 MAE: `0.378062`
  - OOD t2 log10 MAE: `0.426453`
- Phase 6 rerun completed with full stress runs (`primary_variant=proxy_baseline`).
- Phase 7 rerun completed (`multiseed_top_variant=hybrid_full_conf`), source-holdout remains weak due tiny measured set.

Paper claim scope after rerun:
- Supported: reliability-aware ranking, uncertainty gating, and shortlist robustness under shift.
- Not supported: strong absolute coherence prediction claims without substantially more geometry-linked measured data.

## 14) Final Strict Rerun for Paper Outputs (2026-02-22)

Execution profile:
- Full physics generation run with geometry enabled and Palace required:
  - `Dataset/generate_data.py --workdir Dataset --sampling-mode random --geometry-samples 1200 --junction-samples 10 --target-single-rows 10000 --max-pairs 5000 --gmsh-verbosity 0 --mesh-lc-min-um 30 --mesh-lc-max-um 120 --mesh-optimize-threshold 0.45 --no-palace-fallback`
- Public canonical + mapping rebuilt with fitted/model curves excluded by default.
- Upstream artifacts rebuilt for consistency with the regenerated dataset:
  - Phase 1 retrain, Phase 2 NN retrain/validate, Phase 3 validate.
- Downstream reruns completed:
  - Phase 4 train/validate
  - Phase 5 closed-loop
  - Phase 6 full reliability
  - Phase 7 evidence with 10-seed large sweep to `Phase7_Evidence/artifacts_large_sweep`

Strict generation outcomes:
- `Dataset/final_dataset_single.csv`: 4,142 rows.
- `Dataset/final_dataset_coupled.csv`: 80,000 rows.
- Capacitance source: all accepted single rows are Palace-based (`palace_cached`), no fallback errors recorded.
- Geometry artifacts: 1,027 unique referenced GDS files, all existing and non-empty.

Public bootstrap rebuild outcomes:
- Canonical build: `rows_total=1244`.
- After large-source augmentation: `rows_output=1245`.
- Strict mapping (no fitted/model rows): `mapped_rows_final=5`, `rejected_fitted=1240`.

Model/evidence outcomes (this rerun):
- Phase 2 NN:
  - ID retrieval top-1/top-5: `0.3004 / 0.8942`
  - OOD retrieval top-1/top-5: `0.0817 / 0.3033`
- Phase 3 validation:
  - ID freq MAE mean: `0.004040` (baseline `0.006265`)
  - OOD freq MAE mean: `0.003488` (baseline `0.005257`)
- Phase 5 closed loop:
  - `targets=24`, `candidates=192`, `pass_all=39`, `selected=30`
- Phase 6:
  - `primary_variant=proxy_baseline`
- Phase 7 large sweep (10 seeds):
  - Top by mean reliability rank: `hybrid_full_conf` (`rank_mean=1.6`, `top1_count=5`)
  - `proxy_baseline` remains competitive (`rank_mean=3.0`, `top1_count=4`) with lower absolute OOD MAE values in this run.
  - Source holdout remains weak (`SQuADDS`, 5 rows): high holdout errors, so absolute transfer claims remain unsupported.

Figures generated for paper pack:
- Core pack:
  - `figure1_rank_stability.png`
  - `figure2_ood_boxplots.png`
  - `figure3_source_holdout_table.png`
- Extended pack (additional meaningful visuals):
  - `figure4_risk_coverage_curves.png`
  - `figure5_phase5_candidate_tradeoff.png`
  - `figure6_phase5_passrate_by_target.png`
  - `figure7_variant_abstain_gain.png`
  - `figure8_dataset_distributions.png`
- Location: `Phase7_Evidence/artifacts_large_sweep/figures`
- Manifests:
  - `figure_manifest.json`
  - `figure_manifest_extended.json`

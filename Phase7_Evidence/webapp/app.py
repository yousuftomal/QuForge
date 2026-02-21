from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PHASE6_SUMMARY = ROOT / "Phase6_Reliability" / "artifacts" / "phase6_summary.json"
PHASE5_SCRIPT = ROOT / "Phase5_ClosedLoop" / "run_phase5_closed_loop.py"
SINGLE_CSV = ROOT / "Dataset" / "final_dataset_single.csv"
PHASE2_BUNDLE = ROOT / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt"
PHASE1_MODEL = ROOT / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib"
PHASE1_META = ROOT / "Phase1_Surrogate" / "artifacts" / "single_metadata.json"
WEB_RUNS = ROOT / "Phase7_Evidence" / "artifacts" / "web_runs"


def load_phase6_variants() -> List[Dict[str, str]]:
    if not PHASE6_SUMMARY.exists():
        return []
    payload = json.loads(PHASE6_SUMMARY.read_text(encoding="utf-8"))
    rows = payload.get("phase4_variants", [])
    out: List[Dict[str, str]] = []
    for r in rows:
        artifact_dir = Path(str(r.get("artifact_dir", "")))
        bundle = artifact_dir / "phase4_coherence_bundle.joblib"
        if bundle.exists():
            out.append(
                {
                    "variant": str(r.get("variant", "unknown")),
                    "bundle": str(bundle),
                    "rank": str(r.get("reliability_rank", "?")),
                }
            )
    return out


def run_phase5_from_ui(
    target_freq: float,
    target_anh: float,
    target_ej: float,
    target_ec: float,
    target_charge: float,
    phase4_bundle: Path,
    selected_top_n: int,
    random_state: int,
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = WEB_RUNS / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.DataFrame(
        [
            {
                "target_id": "ui_target_001",
                "freq_01_GHz": target_freq,
                "anharmonicity_GHz": target_anh,
                "EJ_GHz": target_ej,
                "EC_GHz": target_ec,
                "charge_sensitivity_GHz": target_charge,
            }
        ]
    )
    targets_csv = run_dir / "targets_ui.csv"
    targets.to_csv(targets_csv, index=False)

    cmd = [
        sys.executable,
        str(PHASE5_SCRIPT),
        "--single-csv",
        str(SINGLE_CSV),
        "--phase2-bundle",
        str(PHASE2_BUNDLE),
        "--phase1-model",
        str(PHASE1_MODEL),
        "--phase1-meta",
        str(PHASE1_META),
        "--phase4-bundle",
        str(phase4_bundle),
        "--targets-csv",
        str(targets_csv),
        "--num-targets",
        "1",
        "--selected-top-n",
        str(selected_top_n),
        "--output-dir",
        str(run_dir),
        "--random-state",
        str(random_state),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (run_dir / "ui_run.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"Phase 5 failed (exit {proc.returncode})\n{tail}")

    return run_dir

st.set_page_config(page_title="Qubit Design Interface", layout="wide")
st.title("Superconducting Qubit Design Interface")
st.caption("Web interface for closed-loop target-to-candidate generation (Phase 7)")

variants = load_phase6_variants()
if not variants:
    st.error("No Phase 6 variants found. Run Phase 6 first.")
    st.stop()

variant_labels = [f"rank {v['rank']} - {v['variant']}" for v in variants]
label_to_bundle = {variant_labels[i]: Path(variants[i]["bundle"]) for i in range(len(variants))}

left, right = st.columns([1, 1])
with left:
    st.subheader("Design Target")
    target_freq = st.slider("Frequency (GHz)", min_value=4.0, max_value=6.5, value=5.0, step=0.01)
    target_anh = st.slider("Anharmonicity (GHz)", min_value=-0.40, max_value=-0.10, value=-0.20, step=0.001)
    target_ej = st.number_input("EJ (GHz)", min_value=1.0, max_value=40.0, value=18.0, step=0.1)
    target_ec = st.number_input("EC (GHz)", min_value=0.05, max_value=2.0, value=0.25, step=0.01)
    target_charge = st.number_input("Charge Sensitivity (GHz)", min_value=0.0, max_value=1.0, value=0.02, step=0.001)

with right:
    st.subheader("Run Settings")
    selected_label = st.selectbox("Phase 4 reliability bundle", options=variant_labels, index=0)
    selected_top_n = st.slider("Selected candidates", min_value=5, max_value=60, value=30, step=1)
    random_state = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
    st.info("This launches Phase 5 under the hood and returns ranked fabrication candidates.")

if st.button("Run Closed-Loop Design", type="primary"):
    phase4_bundle = label_to_bundle[selected_label]
    with st.spinner("Running Phase 5 closed-loop engine..."):
        try:
            run_dir = run_phase5_from_ui(
                target_freq=target_freq,
                target_anh=target_anh,
                target_ej=target_ej,
                target_ec=target_ec,
                target_charge=target_charge,
                phase4_bundle=phase4_bundle,
                selected_top_n=selected_top_n,
                random_state=int(random_state),
            )
            st.session_state["last_run_dir"] = str(run_dir)
            st.success(f"Run completed: {run_dir}")
        except Exception as exc:
            st.error(str(exc))

last_run = st.session_state.get("last_run_dir")
if last_run:
    run_dir = Path(last_run)
    summary_path = run_dir / "phase5_summary.json"
    selected_path = run_dir / "phase5_selected_candidates.csv"
    candidate_path = run_dir / "phase5_candidate_batch.csv"
    log_path = run_dir / "ui_run.log"

    if summary_path.exists():
        st.subheader("Run Summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates", summary.get("candidates_total", "-"))
        c2.metric("Pass-all", summary.get("candidates_pass_all", "-"))
        c3.metric("Selected", summary.get("selected_total", "-"))
        c4.metric("Selected OOD Rate", f"{float(summary.get('metrics', {}).get('selected_ood_rate', 0.0)):.3f}")

    if selected_path.exists():
        st.subheader("Selected Candidates")
        selected_df = pd.read_csv(selected_path)
        st.dataframe(selected_df, use_container_width=True)
        st.download_button(
            label="Download selected CSV",
            data=selected_path.read_bytes(),
            file_name="phase5_selected_candidates.csv",
            mime="text/csv",
        )

    if candidate_path.exists():
        with st.expander("All candidate batch"):
            cand_df = pd.read_csv(candidate_path)
            st.dataframe(cand_df.head(200), use_container_width=True)

    if log_path.exists():
        with st.expander("Run log"):
            st.code(log_path.read_text(encoding="utf-8")[-4000:])

st.divider()
st.subheader("Phase 6 Reliability Snapshot")
rows = []
for v in variants:
    rows.append({"variant": v["variant"], "rank": v["rank"], "bundle": v["bundle"]})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

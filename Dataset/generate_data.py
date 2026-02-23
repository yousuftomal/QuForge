#!/usr/bin/env python3
"""
Dataset generation pipeline for superconducting transmon design.

Implements the DATA.md workflow end-to-end:
geometry -> EM parameters -> quantum properties.

Key reliability features:
- dry-run support (`--dry-run`) for a small sample before full sweeps
- robust Palace output parsing (supports CSV/TXT variants)
- optional analytic fallback when Palace is unavailable (`--skip-palace`)
- strict unit handling for EC/EJ and coherence time normalization
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


# Physical constants (SI)
E_CHARGE_C = 1.602176634e-19
PLANCK_J_S = 6.62607015e-34

_QISKIT_METAL_LOG_FILTER_CONFIGURED = False


class _QiskitMetalFakeJunctionWarningFilter(logging.Filter):
    """
    Suppress known non-blocking Qiskit Metal warning about missing fake junction GDS.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if (
            "Fake_Junctions.GDS" in message
            and getattr(record, "funcName", "") == "_import_junctions_to_one_cell"
        ):
            return False
        return True


def configure_qiskit_metal_logging() -> None:
    global _QISKIT_METAL_LOG_FILTER_CONFIGURED
    if _QISKIT_METAL_LOG_FILTER_CONFIGURED:
        return

    try:
        from qiskit_metal import logger as metal_logger
    except Exception:
        return

    warning_filter = _QiskitMetalFakeJunctionWarningFilter()

    # Apply to the known Qiskit Metal logger and all active loggers/handlers to
    # reliably suppress only this noisy, non-blocking warning line.
    candidate_loggers: List[logging.Logger] = [metal_logger, logging.getLogger()]
    for logger_obj in logging.Logger.manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            candidate_loggers.append(logger_obj)

    for logger_obj in candidate_loggers:
        logger_obj.addFilter(warning_filter)
        for handler in logger_obj.handlers:
            handler.addFilter(warning_filter)

    _QISKIT_METAL_LOG_FILTER_CONFIGURED = True


@dataclass(frozen=True)
class SingleSweepSpace:
    pad_widths_um: Sequence[float] = (300, 400, 500, 600, 700)
    pad_heights_um: Sequence[float] = (300, 400, 500, 600, 700)
    gap_sizes_um: Sequence[float] = (20, 30, 40, 50)
    junction_areas_um2: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class RandomSweepConfig:
    geometry_samples: int = 1200
    junction_samples: int = 10
    seed: int = 1234
    pad_width_range_um: Tuple[float, float] = (250.0, 900.0)
    pad_height_range_um: Tuple[float, float] = (250.0, 900.0)
    gap_range_um: Tuple[float, float] = (10.0, 80.0)
    junction_area_range_um2: Tuple[float, float] = (0.006, 0.08)


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "geometries": root / "geometries",
        "palace_inputs": root / "palace_inputs",
        "palace_outputs": root / "palace_outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_grid_combinations(space: SingleSweepSpace) -> List[Tuple[float, float, float, float]]:
    return list(
        itertools.product(
            space.pad_widths_um,
            space.pad_heights_um,
            space.gap_sizes_um,
            space.junction_areas_um2,
        )
    )


def build_random_combinations(config: RandomSweepConfig) -> List[Tuple[float, float, float, float]]:
    if config.geometry_samples <= 0:
        return []
    if config.junction_samples <= 0:
        return []

    rng = np.random.default_rng(config.seed)
    pw_lo, pw_hi = sorted((float(config.pad_width_range_um[0]), float(config.pad_width_range_um[1])))
    ph_lo, ph_hi = sorted((float(config.pad_height_range_um[0]), float(config.pad_height_range_um[1])))
    gap_lo, gap_hi = sorted((float(config.gap_range_um[0]), float(config.gap_range_um[1])))
    ja_lo, ja_hi = sorted((float(config.junction_area_range_um2[0]), float(config.junction_area_range_um2[1])))

    geometry_triplets: List[Tuple[float, float, float]] = []
    for _ in range(config.geometry_samples):
        geometry_triplets.append(
            (
                round(float(rng.uniform(pw_lo, pw_hi)), 3),
                round(float(rng.uniform(ph_lo, ph_hi)), 3),
                round(float(rng.uniform(gap_lo, gap_hi)), 3),
            )
        )

    if config.junction_samples == 1:
        junction_areas = [round(float(math.sqrt(ja_lo * ja_hi)), 6)]
    else:
        logs = np.linspace(np.log10(max(ja_lo, 1e-9)), np.log10(max(ja_hi, 1e-9)), config.junction_samples)
        junction_areas = [round(float(x), 6) for x in (10 ** logs)]

    return [(pw, ph, gap, ja) for (pw, ph, gap) in geometry_triplets for ja in junction_areas]


def build_single_combinations(
    space: SingleSweepSpace,
    sampling_mode: str,
    random_config: RandomSweepConfig,
) -> List[Tuple[float, float, float, float]]:
    if sampling_mode == "random":
        return build_random_combinations(random_config)
    return build_grid_combinations(space)


def create_transmon_design(
    pad_width_um: float,
    pad_height_um: float,
    gap_um: float,
    junction_area_um2: float,
    design_id: int,
    geometries_dir: Path,
) -> Path:
    """
    Build a transmon layout in Qiskit Metal and export to GDS.

    Uses inductor width as a junction size proxy from junction area.
    """
    try:
        from qiskit_metal import designs
        from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit Metal is required for geometry generation. "
            "Install with: pip install qiskit-metal pyEPR-quantum"
        ) from exc
    configure_qiskit_metal_logging()

    design = designs.DesignPlanar()
    design.chips.main.size.size_x = "2mm"
    design.chips.main.size.size_y = "2mm"

    junction_width_nm = math.sqrt(junction_area_um2) * 1000.0

    TransmonPocket(
        design,
        "Q1",
        options=dict(
            pad_width=f"{pad_width_um}um",
            pad_height=f"{pad_height_um}um",
            pad_gap=f"{gap_um}um",
            inductor_width=f"{junction_width_nm:.0f}nm",
            pos_x="0um",
            pos_y="0um",
        ),
    )
    design.rebuild()

    gds_path = geometries_dir / f"design_{design_id}.gds"

    renderer = getattr(design.renderers, "gds", None)
    if renderer is None:
        try:
            from qiskit_metal.renderers.renderer_gds.gds_renderer import QGDSRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Unable to access a GDS renderer from Qiskit Metal."
            ) from exc
        renderer = QGDSRenderer(design, initiate=True)

    renderer.export_to_gds(str(gds_path))
    return gds_path


def build_parametric_mesh(
    pad_width_um: float,
    pad_height_um: float,
    gap_um: float,
    design_id: int,
    palace_inputs_dir: Path,
    gmsh_verbosity: int = 2,
    mesh_lc_min_um: float = 35.0,
    mesh_lc_max_um: float = 140.0,
    mesh_optimize_threshold: float = 0.35,
) -> Path:
    """
    Build a Palace-compatible 3D electrostatic mesh directly in Gmsh.

    The geometry is a rectangular domain with two rectangular conductor cavities
    (terminals), parameterized by transmon pad dimensions and gap.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("gmsh is required for mesh conversion. Install with: pip install gmsh") from exc

    mesh_path = palace_inputs_dir / f"design_{design_id}.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", float(max(0, min(int(gmsh_verbosity), 5))))

        gmsh.model.add(f"design_{design_id}")
        occ = gmsh.model.occ

        margin_um = 150.0
        domain_thickness_um = 120.0
        cavity_depth_um = 10.0
        pad_z0_um = -0.5 * cavity_depth_um

        domain_x_um = max(1200.0, 2.0 * pad_width_um + gap_um + 2.0 * margin_um)
        domain_y_um = max(1200.0, pad_height_um + 2.0 * margin_um)
        domain_z_um = domain_thickness_um

        domain = occ.addBox(
            -0.5 * domain_x_um,
            -0.5 * domain_y_um,
            -0.5 * domain_z_um,
            domain_x_um,
            domain_y_um,
            domain_z_um,
        )

        left_x0 = -0.5 * gap_um - pad_width_um
        right_x0 = 0.5 * gap_um
        pad_y0 = -0.5 * pad_height_um

        pad1 = occ.addBox(left_x0, pad_y0, pad_z0_um, pad_width_um, pad_height_um, cavity_depth_um)
        pad2 = occ.addBox(right_x0, pad_y0, pad_z0_um, pad_width_um, pad_height_um, cavity_depth_um)

        cut_out, _ = occ.cut([(3, domain)], [(3, pad1), (3, pad2)], removeObject=True, removeTool=True)
        if not cut_out:
            raise RuntimeError("Gmsh cut operation failed to produce domain volume")

        occ.synchronize()

        vol_tags = [tag for dim, tag in cut_out if dim == 3]
        if not vol_tags:
            raise RuntimeError("No 3D volume tags found after Gmsh cut")
        domain_vol = vol_tags[0]

        boundary_surfaces = [tag for dim, tag in gmsh.model.getBoundary([(3, domain_vol)], oriented=False) if dim == 2]

        pad1_xmin = left_x0 - 1e-6
        pad1_xmax = left_x0 + pad_width_um + 1e-6
        pad2_xmin = right_x0 - 1e-6
        pad2_xmax = right_x0 + pad_width_um + 1e-6
        pad_ymin = pad_y0 - 1e-6
        pad_ymax = pad_y0 + pad_height_um + 1e-6
        pad_zmin = pad_z0_um - 1e-6
        pad_zmax = pad_z0_um + cavity_depth_um + 1e-6

        terminal_1_surfaces: List[int] = []
        terminal_2_surfaces: List[int] = []
        ground_surfaces: List[int] = []

        for surf_tag in boundary_surfaces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
            is_pad_like = (
                ymin >= pad_ymin
                and ymax <= pad_ymax
                and zmin >= pad_zmin
                and zmax <= pad_zmax
            )
            if is_pad_like and xmin >= pad1_xmin and xmax <= pad1_xmax:
                terminal_1_surfaces.append(surf_tag)
            elif is_pad_like and xmin >= pad2_xmin and xmax <= pad2_xmax:
                terminal_2_surfaces.append(surf_tag)
            else:
                ground_surfaces.append(surf_tag)

        if not terminal_1_surfaces or not terminal_2_surfaces:
            raise RuntimeError("Failed to classify terminal surfaces in generated Gmsh mesh")

        gmsh.model.addPhysicalGroup(3, [domain_vol], tag=1)
        gmsh.model.addPhysicalGroup(2, ground_surfaces, tag=3)
        gmsh.model.addPhysicalGroup(2, terminal_1_surfaces, tag=4)
        gmsh.model.addPhysicalGroup(2, terminal_2_surfaces, tag=5)

        lc_min = float(min(mesh_lc_min_um, mesh_lc_max_um))
        lc_max = float(max(mesh_lc_min_um, mesh_lc_max_um))

        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", float(mesh_optimize_threshold))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            gmsh.model.mesh.optimize()
        gmsh.write(str(mesh_path))
    finally:
        gmsh.finalize()

    return mesh_path


def write_palace_config(mesh_path: Path, design_id: int, palace_inputs_dir: Path) -> Path:
    """
    Write Palace electrostatic config.

    Attribute numbers are mesh-dependent and may need adjustment for your meshing pipeline.
    """
    mesh_for_config = Path("palace_inputs") / mesh_path.name
    config = {
        "Problem": {
            "Type": "Electrostatic",
            "Verbose": 1,
            "Output": f"palace_outputs/design_{design_id}",
        },
        "Model": {"Mesh": str(mesh_for_config).replace("\\", "/"), "L0": 1e-6},
        "Domains": {"Materials": [{"Attributes": [1], "Permittivity": 1.0}]},
        "Boundaries": {
            "Ground": {"Attributes": [3]},
            "Terminal": [
                {"Index": 1, "Attributes": [4]},
                {"Index": 2, "Attributes": [5]},
            ],
        },
        "Solver": {
            "Order": 2,
            "Electrostatic": {"Save": 2},
            "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1.0e-8, "MaxIts": 150},
        },
    }

    config_path = palace_inputs_dir / f"config_{design_id}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _read_numeric_matrix(path: Path) -> np.ndarray:
    """
    Read a numeric matrix from either CSV or whitespace-delimited text.
    """
    data = np.genfromtxt(path, delimiter=",")
    if np.isnan(data).all():
        data = np.genfromtxt(path)

    if data.ndim == 0:
        raise ValueError(f"No numeric matrix parsed from {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if np.isnan(data).any():
        # Keep only fully numeric rows/columns.
        valid_rows = ~np.isnan(data).any(axis=1)
        data = data[valid_rows]
        if data.size == 0:
            raise ValueError(f"Matrix in {path} contains no fully numeric rows")

    # Palace CSV matrices often include a leading index column `i`.
    if data.ndim == 2 and data.shape[1] == data.shape[0] + 1:
        first_col = data[:, 0]
        expected = np.arange(1, data.shape[0] + 1, dtype=float)
        if np.allclose(first_col, expected, rtol=1e-6, atol=1e-9):
            data = data[:, 1:]

    return data.astype(float)


def _find_capacitance_matrix_file(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "terminal-C.csv",
        output_dir / "capacitance_matrix.txt",
        output_dir / "terminal_C.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for file_path in output_dir.rglob("*"):
        lower_name = file_path.name.lower()
        if file_path.is_file() and ("terminal-c" in lower_name or "capacitance" in lower_name):
            return file_path
    return None


def run_palace(config_path: Path, design_id: int, palace_outputs_dir: Path, mount_root: Path) -> np.ndarray:
    """
    Run Palace through Docker and return the capacitance matrix.
    """
    output_dir = palace_outputs_dir / f"design_{design_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_path = mount_root.resolve()
    config_rel = config_path.resolve().relative_to(workspace_path)
    output_rel = output_dir.resolve().relative_to(workspace_path)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace_path}:/workspace",
        "-w",
        "/workspace",
        "ghcr.io/awslabs/palace:latest",
        "palace",
        str(config_rel).replace("\\", "/"),
        "-output",
        str(output_rel).replace("\\", "/"),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    matrix_file = _find_capacitance_matrix_file(output_dir)
    if matrix_file is None:
        raise FileNotFoundError(
            f"Could not locate a capacitance matrix under {output_dir} after Palace run."
        )

    return _read_numeric_matrix(matrix_file)


def docker_image_present(docker_cmd: str, image: str) -> bool:
    try:
        subprocess.run(
            [docker_cmd, "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return True
    except Exception:
        return False


def docker_pull_image(docker_cmd: str, image: str) -> bool:
    try:
        subprocess.run(
            [docker_cmd, "pull", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return True
    except Exception:
        return False


def resolve_palace_image(docker_cmd: str, preferred_image: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if preferred_image:
        candidates.append(preferred_image)
    env_image = os.getenv("PALACE_DOCKER_IMAGE")
    if env_image and env_image not in candidates:
        candidates.append(env_image)
    for default_image in ("ghcr.io/awslabs/palace:latest", "benvial/palace:latest"):
        if default_image not in candidates:
            candidates.append(default_image)

    for image in candidates:
        if docker_image_present(docker_cmd, image) or docker_pull_image(docker_cmd, image):
            return image
    return None


def run_palace_with_image(
    docker_cmd: str,
    palace_image: str,
    config_path: Path,
    design_id: int,
    palace_outputs_dir: Path,
    mount_root: Path,
) -> np.ndarray:
    output_dir = palace_outputs_dir / f"design_{design_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_path = mount_root.resolve()
    config_rel = config_path.resolve().relative_to(workspace_path)
    output_rel = output_dir.resolve().relative_to(workspace_path)

    cmd = [
        docker_cmd,
        "run",
        "--rm",
        "-v",
        f"{workspace_path}:/workspace",
        "-w",
        "/workspace",
        palace_image,
        "palace",
        str(config_rel).replace("\\", "/"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"Palace failed for design {design_id}: {detail}") from exc

    matrix_file = _find_capacitance_matrix_file(output_dir)
    if matrix_file is None:
        raise FileNotFoundError(
            f"Could not locate a capacitance matrix under {output_dir} after Palace run."
        )

    return _read_numeric_matrix(matrix_file)


def find_docker_executable() -> Optional[str]:
    docker_cmd = shutil.which("docker")
    if docker_cmd:
        return docker_cmd

    common_windows_path = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")
    if common_windows_path.exists():
        return str(common_windows_path)
    return None


def palace_runtime_available() -> bool:
    docker_cmd = find_docker_executable()
    if docker_cmd is None:
        return False
    try:
        subprocess.run(
            [docker_cmd, "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return True
    except Exception:
        return False


def estimate_capacitance_matrix_f(pad_width_um: float, pad_height_um: float, gap_um: float) -> np.ndarray:
    """
    Analytic fallback for capacitance (rough order-of-magnitude model).

    Produces terminal-terminal matrix with diagonal scale in the expected
    transmon range (tens to hundreds of fF) for screening workflows.
    """
    # Parallel-plate-like area scaling with simple gap dependence.
    area_um2 = pad_width_um * pad_height_um
    c_ff = 40.0 + 0.01 * area_um2 / max(gap_um, 1.0)  # fF
    c_f = c_ff * 1e-15
    return np.array([[c_f, -0.25 * c_f], [-0.25 * c_f, c_f]], dtype=float)


def capacitance_to_ec_ghz(c_matrix_f: np.ndarray) -> float:
    """
    Convert capacitance matrix to charging energy EC in GHz.

    EC/h = e^2 / (2 C h), with C in farads.
    """
    if c_matrix_f.ndim != 2 or c_matrix_f.shape[0] != c_matrix_f.shape[1]:
        raise ValueError("Capacitance matrix must be square")
    diagonal = np.diag(c_matrix_f).astype(float)
    c_total = float(np.max(diagonal))
    if not np.isfinite(c_total) or c_total <= 0:
        raise ValueError("Invalid capacitance value from matrix")
    return (E_CHARGE_C**2) / (2.0 * c_total * PLANCK_J_S * 1e9)


def junction_area_to_ej_ghz(junction_area_um2: float, jc_a_per_um2: float = 1e-6) -> float:
    """
    Estimate EJ/h in GHz from junction area and critical current density.

    EJ/h = Ic / (4*pi*e), with Ic = Jc * area.
    """
    if junction_area_um2 <= 0:
        raise ValueError("junction_area_um2 must be positive")
    if jc_a_per_um2 <= 0:
        raise ValueError("jc_a_per_um2 must be positive")
    i_c = jc_a_per_um2 * junction_area_um2
    return i_c / (4.0 * math.pi * E_CHARGE_C * 1e9)


def transmon_f01_approx_ghz(ej_ghz: float, ec_ghz: float) -> float:
    """
    Harmonic transmon approximation: f01 ~ sqrt(8 EJ EC) - EC (all in GHz).
    """
    return math.sqrt(max(8.0 * ej_ghz * ec_ghz, 0.0)) - ec_ghz


def _call_if_supported(obj: Any, method_name: str, kwargs: Dict[str, Any]) -> float:
    method = getattr(obj, method_name, None)
    if method is None:
        return float("nan")

    try:
        signature = inspect.signature(method)
        filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
        value = method(**filtered)
    except Exception:
        return float("nan")

    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_time_to_us(value: float) -> float:
    """
    Normalize coherence times to microseconds.

    Heuristic:
    - if <= 0 or NaN: NaN
    - if < 1e-2, treat as seconds and convert to microseconds
    - else assume already in microseconds
    """
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    if value < 1e-2:
        return value * 1e6
    return value


def _combine_t2_total_us(t1_us: float, tphi_us: float) -> float:
    """
    Compute total dephasing time T2 from T1 and pure dephasing Tphi.

    1/T2 = 1/(2*T1) + 1/Tphi
    """
    t1_ok = np.isfinite(t1_us) and t1_us > 0
    tphi_ok = np.isfinite(tphi_us) and tphi_us > 0

    if t1_ok and tphi_ok:
        denom = (1.0 / (2.0 * t1_us)) + (1.0 / tphi_us)
        if denom <= 0:
            return float("nan")
        return float(1.0 / denom)
    if tphi_ok:
        return float(tphi_us)
    if t1_ok:
        return float(2.0 * t1_us)
    return float("nan")


def compute_quantum_properties(ej_ghz: float, ec_ghz: float) -> Dict[str, float]:
    """
    Compute transmon properties using scqubits.
    """
    try:
        import scqubits as scq
    except ImportError as exc:
        raise RuntimeError("scqubits is required for quantum-property computation.") from exc

    qubit = scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=0.0, ncut=30)

    freq_01 = float(qubit.E01())
    anharm = float(qubit.anharmonicity())
    charge_dispersion = _call_if_supported(qubit, "charge_dispersion", {"i": 0, "j": 1})
    if not np.isfinite(charge_dispersion):
        charge_dispersion = _call_if_supported(qubit, "charge_dispersion", {"i": 0})

    t1_val = _call_if_supported(
        qubit,
        "t1_charge_impedance",
        {"i": 0, "j": 1, "Z": 50.0, "T": 0.015},
    )
    tphi_val = _call_if_supported(qubit, "tphi_1_over_f_ng", {"A_noise": 1e-4})

    t1_us = _normalize_time_to_us(t1_val)
    tphi_us = _normalize_time_to_us(tphi_val)
    t2_us = _combine_t2_total_us(t1_us=t1_us, tphi_us=tphi_us)

    freq_ng_0 = float(scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=0.0, ncut=30).E01())
    freq_ng_025 = float(scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=0.25, ncut=30).E01())

    return {
        "freq_01_GHz": freq_01,
        "anharmonicity_GHz": anharm,
        "EJ_EC_ratio": float(ej_ghz / ec_ghz),
        "charge_dispersion_GHz": charge_dispersion,
        "t1_estimate_us": t1_us,
        "tphi_estimate_us": tphi_us,
        "t2_estimate_us": t2_us,
        "charge_sensitivity_GHz": abs(freq_ng_0 - freq_ng_025),
    }


def compute_coupled_properties(
    ej1_ghz: float,
    ec1_ghz: float,
    ej2_ghz: float,
    ec2_ghz: float,
    g_coupling_ghz: float,
    resonator_freq_ghz: float,
) -> Dict[str, float]:
    """
    Two-transmon + resonator coupled model with scqubits HilbertSpace.

    Dispersive shifts are computed from dressed energies using cross-Kerr
    combinations:
      chi_q1r = E101 - E100 - E001 + E000
      chi_q2r = E011 - E010 - E001 + E000
    and reported as a scalar aggregate chi = mean(chi_q1r, chi_q2r) for
    backward compatibility with existing downstream interfaces.
    """
    try:
        import scqubits as scq
    except ImportError as exc:
        raise RuntimeError("scqubits is required for coupled-system computation.") from exc

    q1 = scq.Transmon(EJ=ej1_ghz, EC=ec1_ghz, ng=0.0, ncut=25)
    q2 = scq.Transmon(EJ=ej2_ghz, EC=ec2_ghz, ng=0.0, ncut=25)
    res = scq.Oscillator(E_osc=resonator_freq_ghz, truncated_dim=6)

    hs = scq.HilbertSpace([q1, q2, res])

    # Use callable simple interface for broad compatibility across scqubits versions.
    hs.add_interaction(g=g_coupling_ghz, op1=q1.n_operator, op2=res.annihilation_operator, add_hc=True)
    hs.add_interaction(
        g=0.8 * g_coupling_ghz,
        op1=q2.n_operator,
        op2=res.annihilation_operator,
        add_hc=True,
    )

    def _analytic_chi(g_ghz: float, wq_ghz: float, wr_ghz: float, alpha_ghz: float) -> float:
        delta = float(wq_ghz - wr_ghz)
        # Keep a small floor to avoid singular behavior near resonance.
        floor = 1e-9
        if abs(delta) < floor or abs(delta + alpha_ghz) < floor:
            return float("nan")
        denom = float(delta * (delta + alpha_ghz))
        if abs(denom) < floor:
            return float("nan")
        return float(-(g_ghz ** 2) * alpha_ghz / denom)

    def _mean_finite(values: Sequence[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.mean(finite))

    g_q1r = float(g_coupling_ghz)
    g_q2r = float(0.8 * g_coupling_ghz)

    # Analytic dispersive estimate used as a robust fallback when dressed-state
    # lookup is unavailable/ambiguous.
    q1_freq_bare = float(q1.E01())
    q2_freq_bare = float(q2.E01())
    q1_anharm = float(q1.anharmonicity())
    q2_anharm = float(q2.anharmonicity())
    chi_q1r_analytic = _analytic_chi(g_q1r, q1_freq_bare, float(resonator_freq_ghz), q1_anharm)
    chi_q2r_analytic = _analytic_chi(g_q2r, q2_freq_bare, float(resonator_freq_ghz), q2_anharm)

    # Numerical dressed-energy extraction (preferred).
    dressed_q1 = float("nan")
    dressed_q2 = float("nan")
    chi_q1r = float("nan")
    chi_q2r = float("nan")
    try:
        # Bare-state labeling gives stable energy combinations for chi extraction.
        hs.generate_lookup(ordering="BE", BEs_count=40)
        e000 = float(hs.energy_by_bare_index((0, 0, 0), subtract_ground=False))
        e100 = float(hs.energy_by_bare_index((1, 0, 0), subtract_ground=False))
        e010 = float(hs.energy_by_bare_index((0, 1, 0), subtract_ground=False))
        e001 = float(hs.energy_by_bare_index((0, 0, 1), subtract_ground=False))
        e101 = float(hs.energy_by_bare_index((1, 0, 1), subtract_ground=False))
        e011 = float(hs.energy_by_bare_index((0, 1, 1), subtract_ground=False))

        dressed_q1 = float(e100 - e000)
        dressed_q2 = float(e010 - e000)
        chi_q1r = float(e101 - e100 - e001 + e000)
        chi_q2r = float(e011 - e010 - e001 + e000)
    except Exception:
        # Keep analytic fallback values below.
        pass

    if not np.isfinite(dressed_q1) or not np.isfinite(dressed_q2):
        evals = np.array(hs.eigenvals(evals_count=10), dtype=float)
        if evals.size < 3:
            raise ValueError("Not enough eigenvalues returned from coupled system")
        dressed_q1 = float(evals[1] - evals[0])
        dressed_q2 = float(evals[2] - evals[0])

    if not np.isfinite(chi_q1r):
        chi_q1r = chi_q1r_analytic
    if not np.isfinite(chi_q2r):
        chi_q2r = chi_q2r_analytic
    chi = _mean_finite([chi_q1r, chi_q2r])

    return {
        "dressed_freq_q1_GHz": dressed_q1,
        "dressed_freq_q2_GHz": dressed_q2,
        "dispersive_shift_chi_q1r_GHz": float(chi_q1r),
        "dispersive_shift_chi_q2r_GHz": float(chi_q2r),
        "dispersive_shift_chi_GHz": chi,
        "coupling_strength_g_GHz": g_coupling_ghz,
    }


def generate_single_dataset(
    workdir: Path,
    max_designs: Optional[int],
    skip_geometry: bool,
    skip_palace: bool,
    palace_fallback: bool,
    palace_image: Optional[str],
    sampling_mode: str,
    random_sweep_config: RandomSweepConfig,
    target_single_rows: Optional[int],
    gmsh_verbosity: int,
    mesh_lc_min_um: float,
    mesh_lc_max_um: float,
    mesh_optimize_threshold: float,
    dry_run: bool,
) -> pd.DataFrame:
    dirs = ensure_dirs(workdir)
    space = SingleSweepSpace()
    all_combinations = build_single_combinations(space, sampling_mode=sampling_mode, random_config=random_sweep_config)
    if dry_run and max_designs is None:
        max_designs = 5
    if max_designs is not None:
        all_combinations = all_combinations[: max(0, max_designs)]

    records: List[Dict[str, Any]] = []
    capacitance_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
    geometry_gds_cache: Dict[Tuple[float, float, float], Path] = {}
    docker_cmd = find_docker_executable()
    palace_enabled = not skip_palace and docker_cmd is not None and palace_runtime_available()
    resolved_palace_image: Optional[str] = None
    if palace_enabled and docker_cmd is not None:
        resolved_palace_image = resolve_palace_image(docker_cmd, palace_image)
        palace_enabled = resolved_palace_image is not None
    if not skip_palace and not palace_enabled:
        print("Palace runtime not available (Docker unavailable or not running); using analytic capacitance fallback.")
    elif resolved_palace_image:
        print(f"Using Palace image: {resolved_palace_image}")

    accepted_rows = 0
    for design_id, (pw, ph, gap, ja) in enumerate(tqdm(all_combinations, desc="single")):
        base_record = {
            "design_id": design_id,
            "pad_width_um": pw,
            "pad_height_um": ph,
            "gap_um": gap,
            "junction_area_um2": ja,
        }
        try:
            cache_key = (float(pw), float(ph), float(gap))
            if skip_geometry:
                gds_path = dirs["geometries"] / f"design_{design_id}.gds"
                gds_path.touch(exist_ok=True)
            else:
                if cache_key in geometry_gds_cache:
                    gds_path = geometry_gds_cache[cache_key]
                else:
                    gds_path = create_transmon_design(pw, ph, gap, ja, design_id, dirs["geometries"])
                    geometry_gds_cache[cache_key] = gds_path

            cap_source = "analytic_fallback"
            palace_error = ""
            if cache_key in capacitance_cache:
                c_matrix = capacitance_cache[cache_key]
                cap_source = "palace_cached" if palace_enabled else "analytic_fallback_cached"
            elif palace_enabled:
                try:
                    mesh_path = build_parametric_mesh(
                        pad_width_um=pw,
                        pad_height_um=ph,
                        gap_um=gap,
                        design_id=design_id,
                        palace_inputs_dir=dirs["palace_inputs"],
                        gmsh_verbosity=gmsh_verbosity,
                        mesh_lc_min_um=mesh_lc_min_um,
                        mesh_lc_max_um=mesh_lc_max_um,
                        mesh_optimize_threshold=mesh_optimize_threshold,
                    )
                    config_path = write_palace_config(mesh_path, design_id, dirs["palace_inputs"])
                    c_matrix = run_palace_with_image(
                        docker_cmd=docker_cmd or "docker",
                        palace_image=resolved_palace_image or "ghcr.io/awslabs/palace:latest",
                        config_path=config_path,
                        design_id=design_id,
                        palace_outputs_dir=dirs["palace_outputs"],
                        mount_root=workdir,
                    )
                    cap_source = "palace"
                    capacitance_cache[cache_key] = c_matrix
                except Exception as palace_exc:
                    if not palace_fallback:
                        raise
                    palace_error = str(palace_exc)
                    c_matrix = estimate_capacitance_matrix_f(pw, ph, gap)
                    capacitance_cache[cache_key] = c_matrix
            else:
                c_matrix = estimate_capacitance_matrix_f(pw, ph, gap)
                capacitance_cache[cache_key] = c_matrix

            ec_ghz = capacitance_to_ec_ghz(c_matrix)
            ej_ghz = junction_area_to_ej_ghz(ja)

            ratio = ej_ghz / ec_ghz
            if ratio < 10.0 or ratio > 150.0:
                continue

            approx_f01 = transmon_f01_approx_ghz(ej_ghz, ec_ghz)
            if not (3.0 <= approx_f01 <= 8.0):
                continue

            qprops = compute_quantum_properties(ej_ghz, ec_ghz)

            records.append(
                {
                    **base_record,
                    "gds_path": str(gds_path),
                    "capacitance_source": cap_source,
                    "palace_error": palace_error,
                    "EJ_GHz": ej_ghz,
                    "EC_GHz": ec_ghz,
                    **qprops,
                }
            )
            accepted_rows += 1
            if target_single_rows is not None and accepted_rows >= target_single_rows:
                break
        except Exception as exc:
            records.append({**base_record, "error": str(exc)})

    return pd.DataFrame(records)


def generate_coupled_dataset(single_df: pd.DataFrame, max_pairs: int, dry_run: bool) -> pd.DataFrame:
    if single_df.empty:
        return pd.DataFrame()

    usable = single_df.copy()
    usable = usable[usable.get("error").isna()] if "error" in usable.columns else usable
    if usable.empty:
        return pd.DataFrame()
    required_cols = {"freq_01_GHz", "EJ_GHz", "EC_GHz", "design_id"}
    if not required_cols.issubset(set(usable.columns)):
        return pd.DataFrame()

    good = usable[(usable["freq_01_GHz"] > 4.0) & (usable["freq_01_GHz"] < 6.0)]
    sampled = good.iloc[::5].reset_index(drop=True)
    if sampled.empty:
        sampled = usable.head(10).reset_index(drop=True)

    pair_indices = list(itertools.combinations(range(len(sampled)), 2))
    if dry_run:
        max_pairs = min(max_pairs, 10)
    pair_indices = pair_indices[:max_pairs]

    g_values = [0.05, 0.10, 0.15, 0.20]
    resonator_values = [6.0, 6.5, 7.0, 7.5]

    rows: List[Dict[str, Any]] = []
    for idx1, idx2 in tqdm(pair_indices, desc="coupled"):
        d1 = sampled.iloc[idx1]
        d2 = sampled.iloc[idx2]
        for g in g_values:
            for res_freq in resonator_values:
                try:
                    cprops = compute_coupled_properties(
                        float(d1["EJ_GHz"]),
                        float(d1["EC_GHz"]),
                        float(d2["EJ_GHz"]),
                        float(d2["EC_GHz"]),
                        g,
                        res_freq,
                    )
                    rows.append(
                        {
                            "design_id_1": int(d1["design_id"]),
                            "design_id_2": int(d2["design_id"]),
                            "freq_q1_GHz": float(d1["freq_01_GHz"]),
                            "freq_q2_GHz": float(d2["freq_01_GHz"]),
                            "resonator_freq_GHz": res_freq,
                            **cprops,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "design_id_1": int(d1["design_id"]),
                            "design_id_2": int(d2["design_id"]),
                            "resonator_freq_GHz": res_freq,
                            "coupling_strength_g_GHz": g,
                            "error": str(exc),
                        }
                    )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate superconducting qubit datasets")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Directory where dataset artifacts are written",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Limit number of single-qubit sweep points",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        help="Limit number of sampled qubit pairs for coupled sweep",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("grid", "random"),
        default="grid",
        help="Single-qubit sweep strategy: full grid or random coverage.",
    )
    parser.add_argument(
        "--geometry-samples",
        type=int,
        default=1200,
        help="Number of random geometry triplets (used when --sampling-mode=random).",
    )
    parser.add_argument(
        "--junction-samples",
        type=int,
        default=10,
        help="Number of sampled junction areas (used when --sampling-mode=random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for random sweep sampling.",
    )
    parser.add_argument(
        "--target-single-rows",
        type=int,
        default=None,
        help="Stop single sweep once this many valid single rows are collected.",
    )
    parser.add_argument(
        "--pad-width-range-um",
        type=float,
        nargs=2,
        default=(250.0, 900.0),
        metavar=("MIN", "MAX"),
        help="Pad width range for random sampling in microns.",
    )
    parser.add_argument(
        "--pad-height-range-um",
        type=float,
        nargs=2,
        default=(250.0, 900.0),
        metavar=("MIN", "MAX"),
        help="Pad height range for random sampling in microns.",
    )
    parser.add_argument(
        "--gap-range-um",
        type=float,
        nargs=2,
        default=(10.0, 80.0),
        metavar=("MIN", "MAX"),
        help="Pad gap range for random sampling in microns.",
    )
    parser.add_argument(
        "--junction-area-range-um2",
        type=float,
        nargs=2,
        default=(0.006, 0.08),
        metavar=("MIN", "MAX"),
        help="Junction area range for random sampling in um^2.",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip Qiskit Metal geometry generation (creates placeholders)",
    )
    parser.add_argument(
        "--skip-palace",
        action="store_true",
        help="Skip Palace and use an analytic capacitance estimate",
    )
    parser.add_argument(
        "--no-palace-fallback",
        action="store_true",
        help="Fail instead of using analytic fallback if Palace run fails",
    )
    parser.add_argument(
        "--palace-image",
        type=str,
        default=None,
        help="Override Palace Docker image (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a short validation pass (defaults to 5 designs)",
    )
    parser.add_argument(
        "--gmsh-verbosity",
        type=int,
        default=1,
        help="Gmsh log verbosity (0-5). Default 1 prints errors only.",
    )
    parser.add_argument(
        "--mesh-lc-min-um",
        type=float,
        default=35.0,
        help="Minimum Gmsh characteristic length in microns.",
    )
    parser.add_argument(
        "--mesh-lc-max-um",
        type=float,
        default=140.0,
        help="Maximum Gmsh characteristic length in microns.",
    )
    parser.add_argument(
        "--mesh-optimize-threshold",
        type=float,
        default=0.35,
        help="Gmsh optimization quality threshold (0-1). Higher tightens quality cleanup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if args.mesh_lc_min_um <= 0 or args.mesh_lc_max_um <= 0:
        raise SystemExit("mesh-lc-min-um and mesh-lc-max-um must be > 0")
    if not (0.0 <= args.mesh_optimize_threshold <= 1.0):
        raise SystemExit("mesh-optimize-threshold must be between 0 and 1")
    if args.geometry_samples <= 0 or args.junction_samples <= 0:
        raise SystemExit("geometry-samples and junction-samples must be > 0")
    if args.target_single_rows is not None and args.target_single_rows <= 0:
        raise SystemExit("target-single-rows must be > 0")

    random_sweep_config = RandomSweepConfig(
        geometry_samples=args.geometry_samples,
        junction_samples=args.junction_samples,
        seed=args.seed,
        pad_width_range_um=(float(args.pad_width_range_um[0]), float(args.pad_width_range_um[1])),
        pad_height_range_um=(float(args.pad_height_range_um[0]), float(args.pad_height_range_um[1])),
        gap_range_um=(float(args.gap_range_um[0]), float(args.gap_range_um[1])),
        junction_area_range_um2=(
            float(args.junction_area_range_um2[0]),
            float(args.junction_area_range_um2[1]),
        ),
    )

    single_df = generate_single_dataset(
        workdir=workdir,
        max_designs=args.max_designs,
        skip_geometry=args.skip_geometry,
        skip_palace=args.skip_palace,
        palace_fallback=not args.no_palace_fallback,
        palace_image=args.palace_image,
        sampling_mode=args.sampling_mode,
        random_sweep_config=random_sweep_config,
        target_single_rows=args.target_single_rows,
        gmsh_verbosity=args.gmsh_verbosity,
        mesh_lc_min_um=args.mesh_lc_min_um,
        mesh_lc_max_um=args.mesh_lc_max_um,
        mesh_optimize_threshold=args.mesh_optimize_threshold,
        dry_run=args.dry_run,
    )
    single_out = workdir / "final_dataset_single.csv"
    single_df.to_csv(single_out, index=False)

    coupled_df = generate_coupled_dataset(
        single_df=single_df,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
    )
    coupled_out = workdir / "final_dataset_coupled.csv"
    coupled_df.to_csv(coupled_out, index=False)

    print("=== DONE ===")
    print(f"Single dataset rows:  {len(single_df)} -> {single_out}")
    print(f"Coupled dataset rows: {len(coupled_df)} -> {coupled_out}")
    if "error" in single_df.columns:
        print(f"Single rows with errors: {int(single_df['error'].notna().sum())}")
    if not coupled_df.empty and "error" in coupled_df.columns:
        print(f"Coupled rows with errors: {int(coupled_df['error'].notna().sum())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

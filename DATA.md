# Complete Data Generation Guide — All at Once

## What the Complete Dataset Needs to Cover

You need three levels of data, all connected:

```
Physical Geometry → EM Parameters → Quantum Properties
(Qiskit Metal)       (Palace)         (scQubits)
```

Your final dataset row should look like:

**Geometry inputs** (pad size, gap, junction area) → **EM outputs** (capacitance, frequency) → **Quantum outputs** (freq_01, anharmonicity, T1 estimate, T2 estimate)

---

## Step 1: Set Up Your Environment

```bash
# Core tools
pip install qiskit-metal scqubits numpy pandas matplotlib h5py tqdm

# Qiskit Metal also needs
pip install pyEPR-quantum

# Palace via Docker (recommended)
docker pull ghcr.io/awslabs/palace:latest
```

Create your project folder:

```
qubit_dataset/
├── geometries/          # GDS files from Qiskit Metal
├── palace_inputs/       # Mesh + config files for Palace
├── palace_outputs/      # Capacitance matrices from Palace
├── final_dataset.csv    # Everything combined
└── generate_data.py     # Your master script
```

---

## Step 2: Define Your Sweep Space

These are all the variables you will sweep over. This covers single qubits, coupled qubits, and resonator-coupled systems.

```python
import numpy as np
import itertools

# --- Single Transmon Parameters ---
pad_widths_um    = [300, 400, 500, 600, 700]      # capacitor pad width
pad_heights_um   = [300, 400, 500, 600, 700]      # capacitor pad height
gap_sizes_um     = [20, 30, 40, 50]               # gap between pads
junction_areas   = [0.01, 0.02, 0.03, 0.04, 0.05] # in um^2

# --- Coupled System Parameters ---
coupling_gaps_um = [10, 15, 20, 30]               # gap between qubit and resonator
resonator_lengths_um = [4000, 4500, 5000, 5500]   # sets resonator frequency

# --- Charge offset sweep (for charge sensitivity data) ---
ng_values = np.linspace(0, 0.5, 5)               # charge offset

# Total single-qubit combinations
single_combinations = list(itertools.product(
    pad_widths_um, pad_heights_um, gap_sizes_um, junction_areas
))
print(f"Single qubit designs: {len(single_combinations)}")  # 500
```

---

## Step 3: Generate Geometries with Qiskit Metal

```python
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.tlines.meandered_path import RouteMeander
import os

def create_transmon_design(pad_width, pad_height, gap, junction_area, design_id):
    """
    Creates a transmon layout and exports it for Palace simulation.
    Returns the path to the exported file.
    """
    design = designs.DesignPlanar()
    design.chips.main.size.size_x = '2mm'
    design.chips.main.size.size_y = '2mm'

    # Create the transmon qubit
    q1 = TransmonPocket(design, 'Q1', options=dict(
        pad_width      = f'{pad_width}um',
        pad_height     = f'{pad_height}um',
        pad_gap        = f'{gap}um',
        inductor_width = f'{np.sqrt(junction_area) * 1000:.0f}nm',  # junction width proxy
        pos_x          = '0um',
        pos_y          = '0um'
    ))

    design.rebuild()

    # Export to GDS (geometry file)
    gds_path = f'geometries/design_{design_id}.gds'
    design.renderers.gds.export_to_gds(gds_path)

    return gds_path, design

# Run the sweep
gds_files = []
for i, (pw, ph, gap, ja) in enumerate(single_combinations):
    path, _ = create_transmon_design(pw, ph, gap, ja, i)
    gds_files.append({
        'design_id': i,
        'pad_width': pw,
        'pad_height': ph,
        'gap': gap,
        'junction_area': ja,
        'gds_path': path
    })
    if i % 50 == 0:
        print(f"Generated {i}/{len(single_combinations)} geometries")
```

---

## Step 4: Run Palace EM Simulation

Palace takes a mesh file and returns a capacitance matrix. You run it on each geometry.

**First, convert GDS to mesh using Gmsh:**

```python
import subprocess

def gds_to_mesh(gds_path, design_id):
    """Convert GDS geometry to mesh file that Palace can read."""
    mesh_path = f'palace_inputs/design_{design_id}.msh'
    
    # Gmsh conversion (install: pip install gmsh)
    import gmsh
    gmsh.initialize()
    gmsh.merge(gds_path)
    gmsh.model.mesh.generate(2)  # 2D surface mesh
    gmsh.write(mesh_path)
    gmsh.finalize()
    
    return mesh_path
```

**Then write the Palace config file:**

```python
import json

def write_palace_config(mesh_path, design_id):
    """Write the JSON config file Palace needs."""
    config = {
        "Problem": {
            "Type": "Electrostatic",
            "Verbose": 1
        },
        "Model": {
            "Mesh": mesh_path,
            "L0": 1e-6  # units in microns
        },
        "Domains": {
            "Materials": [
                {"Attributes": [1], "Permittivity": 9.8},  # sapphire substrate
                {"Attributes": [2], "Permittivity": 1.0}   # vacuum
            ]
        },
        "Boundaries": {
            "Ground": {"Attributes": [3]},
            "Terminal": [
                {"Index": 1, "Attributes": [4]},  # pad 1
                {"Index": 2, "Attributes": [5]}   # pad 2
            ]
        },
        "Solver": {
            "Order": 2,
            "GSLibrary": "SuperLU"
        }
    }
    
    config_path = f'palace_inputs/config_{design_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def run_palace(config_path, design_id):
    """Run Palace and return the capacitance matrix."""
    output_dir = f'palace_outputs/design_{design_id}/'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.getcwd()}:/workspace',
        'ghcr.io/awslabs/palace:latest',
        'palace', f'/workspace/{config_path}',
        '-output', f'/workspace/{output_dir}'
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Read the capacitance matrix output
    C_matrix = np.loadtxt(f'{output_dir}capacitance_matrix.txt')
    
    return C_matrix
```

---

## Step 5: Convert EM Results to Physics Parameters

Palace gives you a capacitance matrix. Now you convert that to EJ and EC for scQubits.

```python
def capacitance_to_EC(C_matrix):
    """
    Convert capacitance matrix to charging energy EC.
    EC = e^2 / (2 * C_total) in GHz
    """
    e = 1.602e-19       # electron charge
    h = 6.626e-34       # Planck constant
    
    C_total = C_matrix[0, 0]  # total capacitance of island in Farads
    EC = (e**2) / (2 * C_total * h * 1e9)  # convert to GHz
    return EC

def junction_area_to_EJ(junction_area_um2):
    """
    Estimate Josephson energy from junction area.
    Based on typical Al/AlOx/Al junction critical current density ~1 uA/um^2
    EJ = (hbar * Ic) / (2e) in GHz
    """
    hbar = 1.055e-34
    e = 1.602e-19
    h = 6.626e-34
    
    Jc = 1e-6           # critical current density in A/um^2 (typical value)
    Ic = Jc * junction_area_um2  # critical current in Amperes... wait
    # correct units
    Ic = (Jc * 1e12) * junction_area_um2 * 1e-12  # A
    
    EJ = (hbar * Ic) / (2 * e * h * 1e9)  # in GHz
    return EJ
```

---

## Step 6: Compute Quantum Properties with scQubits

```python
import scqubits as scq

def compute_quantum_properties(EJ, EC, ng_values=[0.0, 0.25, 0.5]):
    """
    Given EJ and EC, compute all quantum properties we care about.
    """
    results = {}
    
    qubit = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=30)
    
    # Core properties
    results['freq_01_GHz']       = qubit.E01()
    results['anharmonicity_GHz'] = qubit.anharmonicity()
    results['EJ_EC_ratio']       = EJ / EC
    
    # Charge sensitivity (important for coherence)
    results['charge_dispersion'] = qubit.charge_dispersion(0)
    
    # Noise estimates
    results['t1_estimate_us'] = qubit.t1_charge_impedance(
        i=0, j=1, Z=50, T=0.015
    )
    results['t2_estimate_us'] = qubit.tphi_1_over_f_ng(A_noise=1e-4)
    
    # Charge offset sensitivity — how much freq changes with charge noise
    freq_at_0   = scq.Transmon(EJ=EJ, EC=EC, ng=0.0,  ncut=30).E01()
    freq_at_025 = scq.Transmon(EJ=EJ, EC=EC, ng=0.25, ncut=30).E01()
    results['charge_sensitivity'] = abs(freq_at_0 - freq_at_025)
    
    return results
```

---

## Step 7: The Coupled System Dataset

Do not skip this. Your surrogate needs to understand coupling or it is only half useful.

```python
def compute_coupled_properties(EJ1, EC1, EJ2, EC2, g_coupling, resonator_freq):
    """
    Two transmons coupled through a resonator.
    """
    q1 = scq.Transmon(EJ=EJ1, EC=EC1, ng=0, ncut=30)
    q2 = scq.Transmon(EJ=EJ2, EC=EC2, ng=0, ncut=30)
    res = scq.Oscillator(E_osc=resonator_freq, truncated_dim=5)
    
    hs = scq.HilbertSpace([q1, q2, res])
    
    hs.add_interaction(
        g_strength=g_coupling,
        op1=q1.n_operator(),
        op2=res.creation_operator() + res.annihilation_operator()
    )
    hs.add_interaction(
        g_strength=g_coupling * 0.8,  # slightly different for q2
        op1=q2.n_operator(),
        op2=res.creation_operator() + res.annihilation_operator()
    )
    
    hs.generate_lookup()
    eigenvals = hs.eigenvals(evals_count=10)
    
    # Dispersive shift — critical parameter for readout
    chi = eigenvals[2] - eigenvals[1] - resonator_freq
    
    return {
        'dressed_freq_q1': float(eigenvals[1]),
        'dressed_freq_q2': float(eigenvals[2]),
        'dispersive_shift_chi': float(chi),
        'coupling_strength_g': g_coupling
    }
```

---

## Step 8: The Master Script That Does Everything

```python
import pandas as pd
from tqdm import tqdm

all_records = []

print("=== PHASE A: Single Transmon Dataset ===")
for record in tqdm(gds_files):
    try:
        # EM simulation
        mesh_path   = gds_to_mesh(record['gds_path'], record['design_id'])
        config_path = write_palace_config(mesh_path, record['design_id'])
        C_matrix    = run_palace(config_path, record['design_id'])
        
        # Convert to physics parameters
        EC = capacitance_to_EC(C_matrix)
        EJ = junction_area_to_EJ(record['junction_area'])
        
        # Skip unphysical designs
        if EJ / EC < 10 or EJ / EC > 150:
            continue
        if not (3.0 < EC * (EJ/EC)**0.5 < 8.0):  # rough freq filter
            continue
        
        # Quantum properties
        qprops = compute_quantum_properties(EJ, EC)
        
        # Combine everything into one record
        full_record = {**record, 'EJ': EJ, 'EC': EC, **qprops}
        all_records.append(full_record)
        
    except Exception as e:
        print(f"Failed on design {record['design_id']}: {e}")
        continue

print(f"\nSingle qubit records: {len(all_records)}")

print("\n=== PHASE B: Coupled System Dataset ===")
coupled_records = []

# Sample a subset of good single-qubit designs for coupling sweep
good_designs = [r for r in all_records if 4.0 < r['freq_01_GHz'] < 6.0]
sampled = good_designs[::5]  # every 5th one

for d1, d2 in tqdm(list(itertools.combinations(sampled[:30], 2))):
    for g in [0.05, 0.10, 0.15, 0.20]:
        for res_freq in [6.0, 6.5, 7.0, 7.5]:
            try:
                cprops = compute_coupled_properties(
                    d1['EJ'], d1['EC'],
                    d2['EJ'], d2['EC'],
                    g, res_freq
                )
                coupled_records.append({
                    'design_id_1': d1['design_id'],
                    'design_id_2': d2['design_id'],
                    'freq_q1': d1['freq_01_GHz'],
                    'freq_q2': d2['freq_01_GHz'],
                    'resonator_freq': res_freq,
                    **cprops
                })
            except:
                continue

print(f"Coupled system records: {len(coupled_records)}")

# Save both datasets
df_single  = pd.DataFrame(all_records)
df_coupled = pd.DataFrame(coupled_records)

df_single.to_csv('final_dataset_single.csv',  index=False)
df_coupled.to_csv('final_dataset_coupled.csv', index=False)

print("\n=== DONE ===")
print(f"Single qubit dataset:  {len(df_single)} rows")
print(f"Coupled system dataset: {len(df_coupled)} rows")
print("Columns:", list(df_single.columns))
```

---

## What You End Up With

**final_dataset_single.csv** — ~400–500 rows covering:
pad geometry → capacitance → EJ, EC → freq, anharmonicity, charge sensitivity, T1/T2 estimates

**final_dataset_coupled.csv** — ~500–1000 rows covering:
two qubit designs + coupling strength + resonator frequency → dressed frequencies, dispersive shift

---

## One Warning Before You Run This

The Palace + mesh conversion step is the most fragile part. If Palace fails on a design, the try/except block will skip it cleanly. But do a **dry run on 5 designs first** before launching the full sweep. Make sure Palace is returning sensible capacitance matrices (should be in the range of 50–200 femtofarads for a typical transmon pad) before wasting compute on 500 designs.
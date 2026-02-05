# PINN-SPM

**Physics-Informed Neural Networks for Single Particle Model Parameter Identification and State Estimation**

This repository implements a physics-informed learning framework that combines recurrent neural networks (LSTM) with a single particle model (SPM) of a lithium-ion battery. The framework jointly identifies battery model parameters and estimates internal states from current and voltage measurements, with optional transfer learning from a known (e.g., fresh) cell to an unknown (e.g., aged) cell.

---

## Overview

The model consists of:

- **Neural networks:** LSTMs that take historical current and voltage and predict initial conditions for the lithium-ion concentration field in each electrode.
- **Battery model:** An SPM discretized by the finite difference method (FDM) that propagates these states and computes terminal voltage.

Training minimizes both **data loss** (predicted vs. observed voltage) and **physics loss** (consistency with the SPM dynamics), so that the learned parameters and states are physically plausible. After training, the same network serves as a fast state estimator.

For methodological details, see the publication linked in [References](#references). Note: the code uses an LSTM and FDM-based SPM; the paper may use a simpler NN and battery model.

---

## Features

- **Parameter identification:** Up to six parameters can be estimated during training:
  - `nLi` — total lithium in solid phase  
  - `R_f_n` — anode SEI film resistance  
  - `k_n`, `k_p` — reaction rate constants (anode, cathode)  
  - `D_s_n`, `D_s_p` — solid-phase diffusion coefficients (anode, cathode)  

- **State estimation:** Trained model provides rapid estimates of internal states (e.g., surface and average concentrations) from current/voltage sequences.

- **Cell models:** Two parameter sets are included (fresh and aged), based on [SPMeT](https://github.com/scott-moura/SPMeT) (LCO parameters). Electrolyte dynamics are neglected in the current SPM.

- **Driving-cycle data:** Pre-generated SPM-FDM simulation data for multiple protocols (UDDS, FUDS, BJDST, Charge, Charge2, US06, US06 Extended) for both cells.

- **Transfer learning:** Pre-train the LSTM on a cell with known parameters (e.g., fresh), then train the full PINN on the target cell (e.g., aged) for faster and more stable parameter identification.

---

## Requirements

- **Python:** 3.8 or higher (tested with 3.10+)
- **Key dependencies:** PyTorch, NumPy, Pandas, PyYAML, Matplotlib; optional: Weights & Biases (wandb) for experiment tracking

Install with:

```bash
pip install -r requirements.txt
```

(Optional) To use GPU acceleration, install a [CUDA-enabled build of PyTorch](https://pytorch.org/get-started/locally/) appropriate for your system.

---

## Project Structure

```
PINN-SPM/
├── main.py              # Entry point; dispatches to NN or PINN training
├── config_nn.yaml        # Configuration for NN pre-training
├── config_pinn.yaml      # Configuration for PINN training (parameter ID)
├── train_nn.py           # LSTM pre-training (known-parameter cell)
├── train_pinn.py         # PINN training (parameter ID + state estimation)
├── custom_lstm.py        # LSTM architecture
├── spm_fdm.py            # Single particle model (FDM)
├── integrate_spmfdm.py   # Integration of SPM with LSTM outputs
├── integrator.py         # Numerical integrators (e.g., RK4)
├── init_params.py        # Battery parameters and parameter sets
├── helper_func.py        # Data loading and utilities
├── data/                 # SPM simulation data (CSV)
└── results/              # Saved models and outputs
```

---

## Usage

### 1. PINN training (parameter identification)

Train the physics-informed model for the target cell (e.g., aged) using a pre-trained LSTM:

```bash
python main.py config_pinn.yaml
```

- Set **`load_nn: True`** and **`suffix_nn`** in `config_pinn.yaml` to load a pre-trained LSTM.
- Choose which parameters to identify via the **`p_targets`** list in `config_pinn.yaml`.
- Adjust **`train_data_type`** and **`val_data_type`** to select driving cycles.
- Saved PINN weights and optional artifacts go to `results/` (and wandb if enabled).

### 2. NN pre-training (optional transfer learning)

Pre-train the LSTM on the cell with known parameters (e.g., fresh cell):

```bash
python main.py config_nn.yaml
```

- Use **`config_nn.yaml`** for data selection, LSTM size, and training duration.
- After training, set **`load_nn: True`** and **`suffix_nn`** in `config_pinn.yaml` to use this NN when training the PINN.

### 3. Configuration

| File            | Purpose |
|-----------------|--------|
| **config_pinn.yaml** | Method (`pinn`), load/save paths, which parameters to identify (`p_targets`), train/val data, battery and solver settings. |
| **config_nn.yaml**   | Method (`nn`), LSTM/FC layout, data, battery settings (no parameter ID). |

Important options:

- **`cell_known`** / **`cell_target`**: 1 = fresh, 2 = aged; used for transfer learning and which parameter set to use.
- **`p_targets`**: Six booleans corresponding to `[nLi, R_f_n, k_n, k_p, D_s_n, D_s_p]`; `True` means that parameter is identified during PINN training.
- **`n_r`**: Number of radial discretization points in the SPM (must match the data; e.g., 20 or 100).
- **`k`**, **`h`**: Input sequence length and integration step for the LSTM/SPM coupling.

---

## Data

The **`data/`** directory contains CSV files from SPM-FDM simulations:

- Naming: `SPM_FDM_nr{n_r}_simulation_{Cycle}_cell{1|2}.csv`
- Cycles: UDDS, FUDS, BJDST, Charge, Charge2, US06, US06_Extended (and DST where available).
- Columns include: `Test_Time(s)`, `Current(A)`, `Voltage(V)`, and state variables such as `Css_n`, `Css_p`, `Cs_ave_n`, `Cs_ave_p`.

Ensure **`n_r`** in the config matches the data files (e.g., 20 or 100).

---

## References

- **PINN methodology (concept):** [Physics-Informed Neural Networks for Battery Parameter Identification and State Estimation](https://ieeexplore.ieee.org/document/10644822) (IEEE; note: paper may use a simpler NN/SPM than this code).
- **Single particle model and parameters:** [SPMe Observability Paper](https://ecal.studentorg.berkeley.edu/pubs/SPMe-Obs-Journal-Final.pdf) (Scott Moura, UC Berkeley) and [SPMeT repository](https://github.com/scott-moura/SPMeT). Electrolyte dynamics are disabled in this implementation.

---

## License

See the repository for license information.

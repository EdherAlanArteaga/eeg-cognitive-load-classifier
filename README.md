# EEG Cognitive Load Classifier

**Edher Alan Arteaga Marroquin · 2026**

Classifies brain efficiency during mental arithmetic from two 30-second EEG recordings — one at rest, one during task — using the Motor Omega algorithm to extract spectral geometry observables R and C.

**Live demo:** [Deploy link here]

---

## What it does

Upload two EDF files for a subject (rest + arithmetic task). The app:

1. Extracts R (spectral rigidity) and C (dynamic coherence) from 23 EEG channels using the Motor Omega
2. Computes ΔR and ΔC — how much the brain's network reorganized going from rest to task
3. Classifies the subject as **G (Efficient)** or **B (Unstable)**
4. Visualizes the subject's position in SPG space against the reference dataset

---

## The finding

One would expect that someone good at mental arithmetic works harder mentally. The data shows the opposite.

Efficient brains (Group G, average 22 subtractions/min) show lower network reorganization during the task. Unstable brains (Group B, average 6 subtractions/min) show larger Delta C — more effort, less output.

The efficient brain does not need to reorganize its network to face the problem. It is already ready.

---

## Method: Motor Omega

The Motor Omega takes a raw EEG signal and extracts two numbers that summarize the state of the brain's communication network at any moment:

**R — Spectral Rigidity**
How stable the brain network is over time. Low R means information flows freely between nodes. High R means the network is stuck in one state.

**C — Dynamic Coherence**
How synchronized brain regions are with each other. High C means one region dominates. Low C means regions work uniformly.

Together, R and C describe whether the brain is in **explorer mode** (flexible, adapting) or **crystallized mode** (rigid, not adapting).

```python
def motor_omega(sig):
    analytic = hilbert(sig)
    phases = np.angle(analytic)
    amplitudes = np.abs(analytic)
    r_val = np.abs(np.mean(np.exp(1j * phases)))
    norma = np.sqrt(np.sum(amplitudes**2, axis=0))
    c_val = np.mean(amplitudes[0] / (norma + 1e-9))
    return r_val, c_val
```

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | PhysioNet EEG During Mental Arithmetic Tasks |
| Subjects | 36 (26 Group G, 10 Group B) |
| Classifier | Random Forest |
| Accuracy (test set) | 72.7% |
| Monte Carlo (100k iterations) | Mean 78.3%, Max 100% |
| R-C decoupling (Spearman) | r = 0.031, p = 0.857 — INDEPENDENT |
| Features | R_rest, R_task, C_rest, C_task, ΔR, ΔC |

**R and C are statistically independent** — changes in synchronization do not produce proportional changes in coherence. This decoupling is the core validation of the SPG framework.

---

## Dataset

Public dataset from PhysioNet (MIT):

**EEG During Mental Arithmetic Tasks** v1.0.0  
https://physionet.org/content/eegmat/1.0.0/

- 36 subjects, 72 EDF recordings (rest + task per subject)
- 500 Hz sampling rate, 23 channels, ~182 seconds per recording
- Group classification based on `Count quality` column in `subject-info.csv`

Note: PhysioNet reports 24G/12B in its general description but the official `subject-info.csv` indicates 26G/10B. This project uses the CSV as source of truth.

---

## Installation

```bash
pip install streamlit numpy pandas scipy matplotlib pyedflib scikit-learn
streamlit run app.py
```

Or deploy directly on [Streamlit Cloud](https://streamlit.io/cloud) by connecting this repository.

---

## Repository structure

```
eeg-cognitive-load-classifier/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

The EDF files are not included — download them from PhysioNet using the link above.

---

## Theoretical framework

This project applies the **SPG (Spectral Persistence Geometry)** framework to EEG signals. SPG measures the topological organization of brain networks through two observables:

- **λ₂** — global connectivity of the network (how integrated the system is)
- **C_dyn = λ₂² · G_pure** — dynamic coherence (where the system sits between exploring and crystallizing)

The Motor Omega is a practical approximation of these observables that operates directly on the raw EEG signal using the Hilbert transform, without requiring explicit network construction.

SPG has been validated on:
- Mental arithmetic (this project)
- Sleep stages (coma → wake gradient, Kendall τ = 0.929, p = 0.0004)
- Major depressive disorder (F(2,100) = 11.74, p < 0.0001, η²p = 0.190)
- Biological and non-biological networks (C. elegans, food web, jazz, Bitcoin)

---

## Limitations

- 36 subjects is a small dataset for a classifier — results should be interpreted with caution on individual subjects
- The Motor Omega uses broadband signal without frequency filtering — filtered versions (alpha, beta) did not improve classification in this dataset
- The R-C decoupling is statistically confirmed but none of the observables individually predict arithmetic performance significantly (all p > 0.05 vs subtractions/min)
- The classifier generalizes to EEG data with similar characteristics to the PhysioNet EEGMAT dataset (500 Hz, 23 channels, standard 10-20 electrode placement)

---

## Citation

Dataset:  
Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O. (2019). EEG During Mental Arithmetic Tasks. PhysioNet. https://doi.org/10.13026/C2R01H

---

*This project is part of a Data Science portfolio. The SPG framework and Motor Omega are original research by Edher Alan Arteaga Marroquin (Zenodo DOI available).*

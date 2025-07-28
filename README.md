# GeoRoutingAMC  
**A Second‑Order Absorbing Markov Chain for Geographic Routing in Aeronautical Communication Networks**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16533710.svg)](https://doi.org/10.5281/zenodo.16533710)

---

## Overview

**GeoRoutingAMC** implements:

- **Monte Carlo simulation** of geographic greedy routing using k-hop neighborhood (Greedy‑k) across varying node equipage fractions  
- **Second‑order absorbing Markov chain** model of success ratio and hop‑stretch factor for Greedy‑k  
- **Entropy** and **conditional entropy** analyses of routing uncertainty  
- **Distance‑vs‑hop** heatmaps relating distance and hop count to ground station 
- **Comparison** of simulation vs. Markov‑model predictions  

All compute‑intensive routines support HPC/Slurm.  

---

## Repository Layout

- `code/` — Python scripts  
- `results/`  
  - `csv_files/` — CSVs  
  - `figures/`   — PDF plots  
- `README.md` — Usage guide   

---

## Installation

```bash
# 1. Clone
git clone https://github.com/ComNetsHH/geo-routing-amc.git
cd geo-routing-amc

# 2. Create & activate Conda env
conda create -n georouting python=3.10 pandas=2.3.1 scipy=1.15.3 numpy=1.26.4 matplotlib=3.10.0 networkx=3.4.2 seaborn=0.13.2 scikit-learn=1.7.1 -y
conda activate georouting

# 3. Run the full analysis pipeline
python code/py_run_all.py


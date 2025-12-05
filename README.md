# LEO Federated Orbit Error Correction (Real TLE, PyTorch)

Federated learning toolkit for correcting SGP4 orbit prediction errors of low Earth orbit (LEO) satellites using real Two‑Line Element (TLE) histories and an Attention‑LSTM model in the RTN frame. [file:3][file:4][web:12]

## Features

### Real TLE + SGP4 Pipeline
- Automated download of enhanced TLE histories from Space‑Track for multiple LEO satellites (e.g., Starlink). [web:12]
- SGP4/skyfield propagation to generate time‑tagged state vectors from each TLE.
- Construction of Tang‑style 14‑dimensional feature vectors per sample:
  - Time offset
  - Initial position/velocity
  - Propagated reference position/velocity
  - Ballistic coefficient. [file:3]

### RTN Error Dataset (Tang‑style)
- Converts Cartesian states to RTN/UNW frame.
- Computes 3‑D position error between “prediction” and “reference” TLE states.
- Builds labeled dataset: 14‑D input → 3‑D RTN error target.
- 90/10 train–validation split with shuffling.

### Orbital‑Similarity Clustering
- Computes orbital‑element distances (a, i, e) from sampled trajectories.
- Computes DTW distances between ground‑track longitude–latitude sequences.
- Forms combined similarity matrix and hierarchical clusters to define FL clients.

### Federated Attention‑LSTM
- Per‑client Attention‑LSTM models with:
  - Input: 14‑D features
  - Output: 3‑D RTN error
- Cosine‑annealed AdamW optimization with gradient clipping.
- FedAvg aggregation across orbital clusters each round.
- Training history tracking (train vs validation MSE in km² per round).

### Performance Metrics & Plots
- Computes baseline and ML‑corrected RMS error per RTN component.
- Implements Tang’s PM metric per direction and mean PM over [R,T,N]. [file:3]
- Per‑satellite PM table (PM\_T and mean PM).
- Generates:
  - FL training‑progress plots
  - Error‑norm histograms (baseline vs ML‑corrected)
  - PM bar plots vs Tang’s best reported PM
  - Multi‑panel summary figure for paper use.

## Tech Stack

- Python 3 + Jupyter Notebook
- PyTorch (Attention‑LSTM, federated training)
- NumPy, Pandas, Matplotlib
- SGP4, skyfield (orbit propagation from TLE) [web:12]
- SciPy / scikit‑learn (clustering, DTW, metrics)

## Requirements

- Python 3.9+  
- Space‑Track account for TLE access. [web:12]
- Recommended: virtual environment with packages in `requirements.txt`.

## How to Run

1. **Clone and install**
2. **Configure Space‑Track**
- Open `leo_federated_tle.ipynb`.
- In the TLE download cell, set:
  - `spacetrack_username`
  - `spacetrack_password`.

3. **Execute the notebook**
- Launch Jupyter:
  ```
  jupyter notebook leo_federated_tle.ipynb
  ```
- Run all ~32 cells in order:
  - TLE download
  - Trajectory generation
  - Dataset build
  - Clustering + FL training
  - Evaluation + plotting.

4. **Inspect outputs**
- Console:
  - PM per RTN direction and global mean
  - Per‑satellite PM table.
- Files:
  - Trained model weights (`fl_lstm_error_predictor.pth`)
  - Summary figures (`training_progress.png`, `pm_summary.png`, etc.).

## Customization

- **Satellites**: edit the list of NORAD IDs in the first cells.
- **Prediction horizon**: change the max time offset between TLE pairs.
- **Model size**: adjust hidden size, number of LSTM layers, or attention configuration.
- **Federated setup**: modify clustering logic or number of clients for different constellations.

This project serves as a reproducible LEO case study for federated learning–enhanced orbit prediction using public TLE catalogs.

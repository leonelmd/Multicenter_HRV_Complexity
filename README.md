# Multicenter Parkinson's Disease Cardiac Autonomic Complexity Study

This repository contains the analysis code and data for the multicenter study investigating cardiac autonomic complexity in Parkinson's Disease (PD) across three international cohorts (Chile, Spain, Japan).

## Study Overview

We analyze Heart Rate Variability (HRV) and complexity metrics (Multiscale Entropy) from electrocardiogram (ECG) and photoplethysmogram (PPG) signals to identify robust physiological biomarkers for PD.

**Participating Centers:**
1.  **CETRAM (Chile):** Short-term PPG (5 min).
2.  **Cruces University Hospital (Spain):** Short-term Holter ECG extracts (5-15 min).
3.  **Nagoya University (Japan):** Long-term 24h Holter ECG.

## Repository Structure

```
.
├── data/               # Processed CSV files for formatted metrics and metadata
├── figures/            # Generated figures (Figure1 - Figure8)
├── scripts/            # Python analysis and plotting scripts
└── requirements.txt    # Python dependencies
```

## Setup & Installation

1.  **Clone the repository.**
2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Full Analysis Pipeline

To regenerate all figures and analysis outputs from the processed data:

```bash
cd scripts
python run_pipeline.py
```
This script will:
*   Compute necessary statistics.
*   Generate Figures 1 through 8 in the `figures/` directory.

### 2. Generate Presentation

To compile the generated figures into a PowerPoint presentation with detailed academic legends:

```bash
cd scripts
python generate_presentation.py
```
The output file `Multicenter_Study_Figures.pptx` will be saved in the root directory.

## Pipeline Details

The analysis pipeline (`run_pipeline.py`) orchestrates the execution of individual figure generation scripts:
*   `generate_figure1.py`: Study design & demographics.
*   `generate_figure2.py`: Signal archetypes & raw traces.
*   `generate_figure3.py`: Circadian dynamics (Japan only).
*   `generate_figure4.py`: Multiscale entropy comparison.
*   `generate_figure5.py`: Diagnostic performance benchmarking.
*   `generate_figure6.py`: Age-independency validation.
*   `generate_figure7.py`: Deep Learning model benchmarking.
*   `generate_figure8.py`: Feature importance & clinical utility.

## Data Availability

The processed metrics and statistical data required to reproduce the main findings (Figures 3-8) are included in this repository.

**Note on Figure 2 (Raw Signals):**
The raw physiological time-series data (ECG/PPG traces) visualized in Figure 2 are not included in the public repository due to dataset size and privacy considerations. **These data can be shared upon reasonable request.**

## Contact

Developed by the NeuroEng@Usach group 

"""Where everything lives.

Two kinds of input:

  data/           Derived tables shipped with this repository. Every figure is drawn
                  from these alone.
  HRV_DATA_ROOT   The original per-cohort series (beat times / RR intervals) and raw
                  ECG, which are not distributed (see DATA_AVAILABILITY.md). Only the
                  scripts that rebuild data/ need them.

Set the environment variable HRV_DATA_ROOT to the folder that holds the original data,
laid out as below. If it is not set, the folder original_data/ next to this repository is
used.

  $HRV_DATA_ROOT/
    cetram/raw/{Control,PD}/<ID>_raw.txt        raw ECG, 1 kHz (OpenSignals export)
    cetram/beats/{Control,PD}/<ID>_cleaned.csv  beat times    <- scripts/cohorts/cetram_beats.py
    cruces/raw/ppg_signals.mat                  raw finger PPG, 500 Hz
    cruces/rri/<ID>.csv                         RR intervals  <- scripts/cohorts/cruces_rri.py
    nagoya/rri/<ID>_RRi.txt                     RR intervals as exported by the chest strap

The de-identified clinical table (Fig. 9) is read from data/private/, which is never
committed.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
FIGURES = os.path.join(REPO, "figures")
PRIVATE = os.path.join(DATA, "private")

DATA_ROOT = os.path.abspath(os.environ.get("HRV_DATA_ROOT",
                                           os.path.join(REPO, "..", "original_data")))

CETRAM_RAW_ECG = os.path.join(DATA_ROOT, "cetram", "raw")
CETRAM_PEAKS = os.path.join(DATA_ROOT, "cetram", "beats")
CRUCES_PPG = os.path.join(DATA_ROOT, "cruces", "raw", "ppg_signals.mat")
CRUCES_RRI = os.path.join(DATA_ROOT, "cruces", "rri")
NAGOYA_RRI = os.path.join(DATA_ROOT, "nagoya", "rri")
NAGOYA_META = os.path.join(DATA, "japan_metadata.csv")   # shipped: start time and file per subject
CLINICAL_XLSX = os.path.join(PRIVATE, "deidentified_clinical_consolidated.xlsx")


def figdir(name):
    """figures/<name>/, created on demand. <name> is the manuscript figure, e.g. Fig5_parameters."""
    d = os.path.join(FIGURES, name)
    os.makedirs(d, exist_ok=True)
    return d


def have(path):
    return os.path.exists(path)


def require(path, what):
    """Stop with a clear message when an input that is not distributed is missing."""
    if not os.path.exists(path):
        sys.exit(f"\n  Needs {what}, expected at\n    {path}\n"
                 f"  This input is not distributed with the repository; see DATA_AVAILABILITY.md.\n"
                 f"  Set HRV_DATA_ROOT to the folder holding the original data.\n")
    return path

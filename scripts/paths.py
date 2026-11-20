"""Where everything lives.

Two kinds of input:

  data/           Derived tables shipped with this repository. Every figure is drawn
                  from these alone.
  HRV_DATA_ROOT   The original per-cohort series (beat times / RR intervals) and raw
                  ECG, which are not distributed (see DATA_AVAILABILITY.md). Only the
                  scripts that rebuild data/ need them.

Set the environment variable HRV_DATA_ROOT to the folder that holds the original data,
laid out as below. If it is not set, the folder two levels above this repository is
used, which is the layout on the analysis machine.

  $HRV_DATA_ROOT/
    CETRAM/public_release/results/cleaned_detections/{Control,PD}/<ID>_cleaned.csv
    CETRAM/public_release/data/raw/{Control,PD}/<ID>_raw.txt        (raw ECG, 1 kHz)
    Cruces/public_release/data/processed/RRi/<ID>.csv
    Nagoya/public_release/data/processed_rri/<ID>_RRi.txt
    Nagoya/public_release/data/metadata/metadata.csv

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
                                           os.path.join(REPO, "..", "..")))

CETRAM_PEAKS = os.path.join(DATA_ROOT, "CETRAM", "public_release", "results", "cleaned_detections")
CETRAM_RAW_ECG = os.path.join(DATA_ROOT, "CETRAM", "public_release", "data", "raw")
CRUCES_RRI = os.path.join(DATA_ROOT, "Cruces", "public_release", "data", "processed", "RRi")
NAGOYA = os.path.join(DATA_ROOT, "Nagoya", "public_release")
NAGOYA_RRI = os.path.join(NAGOYA, "data", "processed_rri")
NAGOYA_META = os.path.join(NAGOYA, "data", "metadata", "metadata.csv")
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

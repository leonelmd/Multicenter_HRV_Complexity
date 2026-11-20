"""Illustrative ECG strips for Fig. 1: complex (control, blue) vs regular (PD, red).

Synthetic, not patient data. Identical beat morphology in both strips; only the
beat timing differs:
  control - RR = mean + respiratory oscillation + 1/f-like fluctuations
            (structure on many time scales -> "complex");
            variability strongly exaggerated (hand-set RR, 0.56-1.10 s) for legibility,
            plus mild respiratory R-amplitude modulation
  PD      - RR = mean + small white jitter (nearly metronomic -> "less complex")

Outputs (figures/Fig1_overview/elements/):
  ecg_trace_control.svg, ecg_trace_pd.svg, ecg_traces_pair.svg (+ .png previews)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"

BLUE, RED = "#4C72B0", "#C44E52"
FS = 500          # Hz
DUR = 10.0        # s per strip
MEAN_RR = 0.85    # s (~70 bpm)

# (centre s rel. R, width s, amplitude mV) - P, Q, R, S, T
WAVES = [(-0.20, 0.025, 0.12), (-0.030, 0.010, -0.12), (0.0, 0.011, 1.0),
         (0.032, 0.012, -0.25), (0.26, 0.045, 0.28)]


def rr_control(n, rng):
    # Hand-set, deliberately irregular sequence (slow drift + fast alternation),
    # spanning ~0.55-1.10 s so the uneven spacing is obvious at print size.
    base = np.array([0.78, 1.02, 0.64, 0.90, 1.10, 0.70, 0.56, 0.94, 0.82,
                     1.06, 0.62, 0.86, 0.98, 0.68, 0.92])
    return base[:n] if n <= len(base) else np.resize(base, n)


def rr_pd(n, rng):
    return MEAN_RR + rng.normal(0, 0.003, n)          # metronomic


def synth(rr, amp_mod=0.0):
    t = np.arange(0, DUR, 1 / FS)
    peaks = 0.35 + np.concatenate([[0], np.cumsum(rr)])
    peaks = peaks[peaks < DUR + 0.5]
    y = np.zeros_like(t)
    for i, p in enumerate(peaks):
        g = 1 + amp_mod * np.sin(2 * np.pi * i / 3.7)  # respiratory amplitude change
        for c, w, a in WAVES:
            y += g * a * np.exp(-0.5 * ((t - p - c) / w) ** 2)
    y += 0.03 * np.sin(2 * np.pi * 0.15 * t)                         # mild baseline wander
    return t, y, peaks[peaks < DUR]


def draw(ax, t, y, peaks, colour, ticks):
    ax.plot(t, y, color=colour, lw=1.0, solid_joinstyle="round")
    if ticks:
        ax.plot(peaks, np.full_like(peaks, 1.12), "v", ms=3, color=colour, mec="none")
    ax.set_xlim(0, DUR)
    ax.set_ylim(-0.45, 1.25)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "figures", "Fig1_overview", "elements"))
    ap.add_argument("--ticks", action="store_true", help="mark R peaks")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = int(DUR / 0.6) + 2
    strips = {"control": (synth(rr_control(n, rng), amp_mod=0.12), BLUE),
              "pd": (synth(rr_pd(n, rng)), RED)}

    for name, ((t, y, pk), col) in strips.items():
        fig, ax = plt.subplots(figsize=(3.2, 0.55))
        draw(ax, t, y, pk, col, a.ticks)
        fig.subplots_adjust(0, 0, 1, 1)
        for ext in ("svg", "png"):
            fig.savefig(out / f"ecg_trace_{name}.{ext}", dpi=300, transparent=True)
        plt.close(fig)
        rri = np.diff(pk) * 1000
        print(f"{name}: {len(pk)} beats, RR {rri.mean():.0f} ± {rri.std():.0f} ms "
              f"(range {rri.min():.0f}-{rri.max():.0f})")

    fig, axs = plt.subplots(2, 1, figsize=(3.2, 1.25))
    for ax, (name, ((t, y, pk), col)) in zip(axs, strips.items()):
        draw(ax, t, y, pk, col, a.ticks)
    fig.subplots_adjust(0, 0, 1, 1, hspace=0.15)
    for ext in ("svg", "png"):
        fig.savefig(out / f"ecg_traces_pair.{ext}", dpi=300, transparent=True)


if __name__ == "__main__":
    main()

"""Illustrative ECG strips for Fig. 1: complex (control, blue) vs regular (PD, red).

Synthetic, not patient data. Identical beat morphology in both strips; only the
beat timing differs:
  control - RR = mean + respiratory oscillation + 1/f-like fluctuations
            (structure on many time scales -> "complex");
            variability exaggerated (~3x physiological) so it reads at print size
  PD      - RR = mean + small white jitter (nearly metronomic -> "less complex")

Outputs (figures/Figure1/elements/):
  ecg_trace_control.svg, ecg_trace_pd.svg, ecg_traces_pair.svg (+ .png previews)
"""
import argparse
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
    k = np.arange(n)
    rr = MEAN_RR + 0.09 * np.sin(2 * np.pi * k / 4.2 + 0.6)        # respiration
    for f in np.geomspace(1 / 40, 1 / 2.5, 12):                     # 1/f-like
        rr += 0.11 * (f * 4) ** -0.5 / 4 * np.sin(2 * np.pi * f * k + rng.uniform(0, 2 * np.pi))
    return rr + rng.normal(0, 0.035, n)


def rr_pd(n, rng):
    return MEAN_RR + rng.normal(0, 0.006, n)


def synth(rr):
    t = np.arange(0, DUR, 1 / FS)
    peaks = 0.35 + np.concatenate([[0], np.cumsum(rr)])
    peaks = peaks[peaks < DUR + 0.5]
    y = np.zeros_like(t)
    for p in peaks:
        for c, w, a in WAVES:
            y += a * np.exp(-0.5 * ((t - p - c) / w) ** 2)
    y += 0.03 * np.sin(2 * np.pi * 0.15 * t)                         # mild baseline wander
    return t, y, peaks[peaks < DUR]


def draw(ax, t, y, peaks, colour, ticks):
    ax.plot(t, y, color=colour, lw=1.0, solid_joinstyle="round")
    if ticks:
        ax.plot(peaks, np.full_like(peaks, 1.12), "v", ms=3, color=colour, mec="none")
    ax.set_xlim(0, DUR)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/Figure1/elements")
    ap.add_argument("--ticks", action="store_true", help="mark R peaks")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = int(DUR / 0.6) + 2
    strips = {"control": (synth(rr_control(n, rng)), BLUE),
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

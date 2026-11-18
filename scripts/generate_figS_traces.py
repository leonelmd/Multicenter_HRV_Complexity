#!/usr/bin/env python3
"""Supplementary figure: raw-trace review of the CETRAM recordings with SDNN > 100 ms.

One row per recording: six seconds of ECG on a standard grid (left) and the RR-interval series
of the whole recording with that window shaded (right).
  Reference: C028 and P011, the recordings closest to their group's median complexity index
             in the analysed sample (n = 69).
  Excluded:  P004 and C050 (frequent ectopy), P012 (minutes without QRS; P023 is analogous).
  Retained:  P003 (tremor/movement artefact confined to part of the record; C018 is analogous)
             and C033 (clean sinus rhythm with slow drifts of heart rate).
Beat marks are the analysis marks at their true positions: NeuroKit2 ecg_clean + Brammer
detector + Lipponen-Tarvainen correction, as in CETRAM/public_release/scripts/clean_data.py
(the saved detection files rebuild positions from the guarded intervals and cannot be overlaid
on the ECG). Windows: reference and C033 at mid-record; the other four at the time points cited in the
review sheet (P004 596 s, C050 07:37) or at the return of the ECG signal (P012, 283 s) and the
first artefact episode (P003).

Inputs: raw ECG (not released; CETRAM/public_release/data/raw).
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import figstyle as fs

ROOT = os.path.dirname(HERE); HRV = os.path.dirname(os.path.dirname(ROOT))
RAW = os.path.join(HRV, "CETRAM", "public_release", "data", "raw")
OUT = os.path.join(ROOT, "figures", "Appendix")
FS, L = 1000, 6.0

ROWS = [  # subject, group, status, label, window rule
    ("C028", "Control", "reference", "control closest to the group median", "mid"),
    ("P011", "PD", "reference", "PD closest to the group median", "mid"),
    ("P004", "PD", "excluded", "frequent ectopy, wide complexes", 596.0),
    ("C050", "Control", "excluded", "frequent ectopy, premature narrow beats", 457.0),
    ("P012", "PD", "excluded", "no QRS for most of the first ~5 min", 280.0),
    ("P003", "PD", "retained", "artefact in the first ~2 min; marks on noise", 12.0),
    ("C033", "Control", "retained", "slow drifts of heart rate", "mid"),
]
STATUS_COL = {"reference": "0.35", "excluded": "#C44E52", "retained": "#4C72B0"}


def load(sid, grp):
    path = os.path.join(RAW, grp, f"{sid}_raw.txt")
    with open(path) as f:
        f.readline(); meta = list(json.loads(f.readline()[2:]).values())[0]
    x = pd.read_csv(path, sep="\t", skiprows=3, header=None).iloc[:, 5].to_numpy(float)
    mv = x if meta.get("convertedValues", 0) else ((x / 2 ** meta["resolution"][-1]) - 0.5) * 3.3 / 1100 * 1000
    return x, mv


def display(x):
    b, a = butter(2, 0.5 / (FS / 2), "high"); x = filtfilt(b, a, x)
    b, a = iirnotch(50, 30, FS); x = filtfilt(b, a, x)
    b, a = butter(4, 40 / (FS / 2), "low"); return filtfilt(b, a, x)


def marks(raw):
    import neurokit2 as nk
    ec = nk.ecg_clean(raw, sampling_rate=FS)
    r = np.asarray(nk.ecg_peaks(ec, sampling_rate=FS, method="neurokit")[1]["ECG_R_Peaks"], int)
    _, fixed = nk.signal_fixpeaks(r, sampling_rate=FS, iterative=True, method="Kubios", show=False)
    return np.asarray(fixed, int)


def choose(rule, y, pk):
    if isinstance(rule, float):
        return rule
    dur = len(y) / FS; rr = np.diff(pk); t = pk[1:] / FS
    starts = np.arange(10, dur - L - 10, 1.0)
    if rule == "mid":
        return float(round(dur / 2))
    if rule == "ectopy":
        med = pd.Series(rr).rolling(11, center=True, min_periods=3).median().values
        prem = rr < 0.8 * med
        return float(starts[np.argmax([prem[(t >= s) & (t < s + L)].sum() for s in starts])])
    if rule == "onset":   # first window in which the 2-s peak-to-peak amplitude returns
        W = 2 * FS; ptp = np.array([np.ptp(y[i:i + W]) for i in range(0, len(y) - W, W)])
        lo = ptp < 0.2 * np.median(ptp); k = np.argmax(~lo[np.argmax(lo):]) + np.argmax(lo)
        return float(k * 2 - 3)
    if rule == "noise":   # most high-frequency (>15 Hz) power in the first half
        hf = np.abs(np.diff(y))
        sc = [hf[int(s * FS):int((s + L) * FS)].mean() for s in starts if s < dur / 2]
        return float(starts[int(np.argmax(sc))])


def grid(ax, x1, y0, y1):
    for v in np.arange(0, x1 + 1e-9, 0.04):
        maj = abs(v / 0.2 - round(v / 0.2)) < 1e-6
        ax.axvline(v, color="#f2b8b8" if maj else "#fbe3e3", lw=0.5 if maj else 0.25, zorder=0)
    for v in np.arange(np.floor(y0 * 10) / 10, y1, 0.1):
        maj = abs(v * 2 - round(v * 2)) < 1e-6
        ax.axhline(v, color="#f2b8b8" if maj else "#fbe3e3", lw=0.5 if maj else 0.25, zorder=0)
    ax.grid(False)


def main():
    fs.apply()
    fig = plt.figure(figsize=(7.2, 1.18 * len(ROWS) + 0.4))
    gs = fig.add_gridspec(len(ROWS), 2, width_ratios=[3.1, 1.0], hspace=0.55, wspace=0.08)
    for i, (sid, grp, status, label, rule) in enumerate(ROWS):
        raw, mv = load(sid, grp); y = display(mv); pk = marks(raw)
        s = choose(rule, y, pk); a0, a1 = int(s * FS), int((s + L) * FS)
        seg = y[a0:a1]; c = np.median(seg); y0, y1 = c - 1.1, c + 1.3
        ax = fig.add_subplot(gs[i, 0]); grid(ax, L, y0, y1)
        ax.plot(np.arange(a1 - a0) / FS, seg, color="0.1", lw=0.6, zorder=2)
        m = pk[(pk >= a0) & (pk < a1)]
        ax.scatter((m - a0) / FS, np.full(len(m), y1 - 0.18), marker="v", s=9, color="#4C72B0", zorder=3, lw=0)
        ax.set_xlim(-0.35, L); ax.set_ylim(y0, y1); ax.set_yticks([]); ax.set_xticks([])
        ax.axvspan(-0.35, 0, color="white", zorder=1)
        ax.plot([-0.2, -0.2], [y0 + 0.15, y0 + 1.15], color="0.1", lw=1.1, zorder=4)
        if i == 0: ax.text(-0.14, y0 + 0.65, "1 mV", fontsize=6.5, va="center", zorder=4)
        for sp in ax.spines.values(): sp.set_visible(False)
        mm, ss = int(s // 60), s % 60
        ax.text(0.0, 1.03, f"{sid} ({'PD' if grp == 'PD' else 'control'})", transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left")
        ax.text(1.0, 1.03, f"{status}: {label}", transform=ax.transAxes, fontsize=8,
                color=STATUS_COL[status], va="bottom", ha="right")
        ax.text(L, y0 + 0.05, f"{mm:02d}:{ss:04.1f}–{int((s+L)//60):02d}:{(s+L)%60:04.1f}", fontsize=6.5,
                ha="right", va="bottom", color="0.35")
        # tachogram
        bx = fig.add_subplot(gs[i, 1]); rr = np.diff(pk); t = pk[1:] / FS / 60
        bx.plot(t, np.clip(rr, 300, 2000), color="0.25", lw=0.4)
        bx.axvspan(s / 60, (s + L) / 60, color="#DD8452", alpha=0.5, lw=0)
        bx.plot([(s + L / 2) / 60], [1.0], marker="v", ms=4, color="#DD8452", transform=bx.get_xaxis_transform(), clip_on=False)
        bx.set_ylim(300, 2000 if rr.max() > 1600 else max(1300, rr.max() + 80))
        bx.tick_params(labelsize=6.5); bx.set_xlim(0, len(y) / FS / 60)
        if i == len(ROWS) - 1: bx.set_xlabel("time (min)", fontsize=7.5)
        else: bx.set_xticklabels([])
        if i == 0: bx.set_title("RR interval (ms), whole record", fontsize=7.5, pad=9)
        print(f"{sid}: window {s:.0f}-{s+L:.0f} s ({rule}), {len(pk)} marks")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.05)
    print(" ->", fs.save(fig, OUT, "FigureS_traces"))


if __name__ == "__main__":
    main()

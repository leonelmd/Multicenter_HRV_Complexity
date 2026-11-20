#!/usr/bin/env python3
"""Fig. 6 — Where the beat mark falls, and what it does to the entropy curve (CETRAM).

(A) Six seconds of resting single-lead ECG with the marks of both detectors.
(B) Two beats enlarged: the Pan-Tompkins mark on the R apex and on the S wave.
(C) Timing of each Pan-Tompkins mark relative to the Brammer mark, whole recording.
(D) The two RR-interval series of the same recording, superimposed.
(E) Group-mean rcMSE curves (scales 1-5) for both detectors, analysed CETRAM sample.

Detectors exactly as in CETRAM/public_release/scripts/clean_data.py: NeuroKit2 ecg_clean, then
ecg_peaks(method="neurokit") (Brammer) or method="pantompkins1985". Panels A-D use raw detector
marks (before Lipponen-Tarvainen correction) so that the fiducial itself is shown.

The example recording (C041) is illustrative and was chosen because its Pan-Tompkins marks
alternate between the R apex and the S wave (about half fall 15-60 ms late); in the median
recording about a third do. Panel E uses all analysed participants.

Inputs: raw ECG (not released; CETRAM/public_release/data/raw), data/cetram_detector_curves.csv.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import figstyle as fs

ROOT = os.path.dirname(HERE)
from paths import CETRAM_RAW_ECG as RAW, require
OUT = os.path.join(ROOT, "figures", "Fig6_detector")
FS = 1000
EX, EXG = "C041", "Control"
C_B, C_P = "#4C72B0", "#DD8452"


def load_mv(sid, grp):
    import json
    path = os.path.join(RAW, grp, f"{sid}_raw.txt")
    with open(path) as f:
        f.readline(); meta = list(json.loads(f.readline()[2:]).values())[0]
    x = pd.read_csv(path, sep="\t", skiprows=3, header=None).iloc[:, 5].to_numpy(float)
    if meta.get("convertedValues", 0):
        return x, x
    return x, ((x / 2 ** meta["resolution"][-1]) - 0.5) * 3.3 / 1100 * 1000


def display(x):
    b, a = butter(2, 0.5 / (FS / 2), "high"); x = filtfilt(b, a, x)
    b, a = iirnotch(50, 30, FS); x = filtfilt(b, a, x)
    b, a = butter(4, 40 / (FS / 2), "low"); return filtfilt(b, a, x)


def ecg_grid(ax, x0, x1, y0, y1):
    for v in np.arange(np.ceil(x0 / 0.04) * 0.04, x1, 0.04):
        maj = abs(v / 0.2 - round(v / 0.2)) < 1e-6
        ax.axvline(v, color="#f2b8b8" if maj else "#fbe3e3", lw=0.6 if maj else 0.3, zorder=0)
    for v in np.arange(np.floor(y0 * 10) / 10, y1, 0.1):
        maj = abs(v * 2 - round(v * 2)) < 1e-6
        ax.axhline(v, color="#f2b8b8" if maj else "#fbe3e3", lw=0.6 if maj else 0.3, zorder=0)
    ax.grid(False)


def main():
    import neurokit2 as nk
    fs.apply()
    require(RAW, "the raw CETRAM ECG (panels A-D)")
    raw, mv = load_mv(EX, EXG)
    ec = nk.ecg_clean(raw, sampling_rate=FS)
    b = np.asarray(nk.ecg_peaks(ec, sampling_rate=FS, method="neurokit")[1]["ECG_R_Peaks"], int)
    p = np.asarray(nk.ecg_peaks(ec, sampling_rate=FS, method="pantompkins1985")[1]["ECG_R_Peaks"], int)
    y = display(mv)
    j = np.clip(np.searchsorted(b, p), 1, len(b) - 1)
    near = np.where(np.abs(p - b[j - 1]) < np.abs(p - b[j]), b[j - 1], b[j])
    lag = p - near
    late = (lag > 15) & (lag < 60)
    print(f"{EX}: {len(b)} Brammer / {len(p)} Pan-Tompkins marks; {100*late.mean():.0f}% of PT marks 15-60 ms late; "
          f"median late lag {np.median(lag[late]):.0f} ms")

    # window: 6 s with the most even mix of on-R and late PT marks, away from the edges
    best, bs = None, -1
    for s in range(30, int(len(y) / FS) - 40):
        m = (p >= s * FS) & (p < (s + 6) * FS)
        if m.sum() < 5: continue
        fr = late[m].mean(); sc = min(fr, 1 - fr)
        if sc > bs: best, bs = s, sc
    s0, s1 = best * FS, (best + 6) * FS

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], hspace=0.62, wspace=0.42)

    # A: strip
    ax = fig.add_subplot(gs[0, :]); fs.panel(ax, "A", dx=-0.07)
    t = np.arange(s0, s1) / FS - best
    seg = y[s0:s1]; c = np.median(seg); y0, y1 = c - 1.0, c + 1.4
    ecg_grid(ax, 0, 6, y0, y1)
    ax.plot(t, seg, color="0.1", lw=0.7, zorder=2)
    mb = b[(b >= s0) & (b < s1)]; mp = p[(p >= s0) & (p < s1)]
    ax.scatter(mb / FS - best, y[mb] + 0.18, marker="v", s=22, color=C_B, zorder=3, label="Brammer (R apex)")
    ax.scatter(mp / FS - best, y[mp] - 0.18, marker="^", s=22, color=C_P, zorder=3, label="Pan–Tompkins")
    ax.set_xlim(-0.4, 6); ax.set_ylim(y0, y1); ax.set_yticks([]); ax.set_xticks(range(7))
    ax.set_xlabel("time (s)"); fs.despine(ax, left=True)
    ax.legend(loc="upper right", ncol=2, fontsize=7.5, bbox_to_anchor=(1.0, 1.22))
    ax.axvspan(-0.4, 0, color="white", zorder=1)
    ax.plot([-0.25, -0.25], [y0 + 0.2, y0 + 1.2], color="0.1", lw=1.4, zorder=4)
    ax.text(-0.2, y0 + 0.7, "1 mV", fontsize=7, va="center", zorder=4)

    # B: two beats, one PT on R, one PT late
    ax = fig.add_subplot(gs[1, 0]); fs.panel(ax, "B")
    on = [q for q in mp if not late[np.where(p == q)[0][0]]]
    lt = [q for q in mp if late[np.where(p == q)[0][0]]]
    for k, (q, ls) in enumerate([(on[0] if on else mp[0], "-"), (lt[0] if lt else mp[-1], "-")]):
        w = np.arange(q - 120, q + 160)
        ax.plot((w - q) / FS * 1000, y[w] - k * 1.4, color="0.1", lw=0.9)
        r = b[np.argmin(np.abs(b - q))]
        ax.scatter([(r - q) / FS * 1000], [y[r] - k * 1.4 + 0.15], marker="v", s=26, color=C_B, zorder=3)
        ax.scatter([0], [y[q] - k * 1.4 - 0.15], marker="^", s=26, color=C_P, zorder=3)
        ax.text(150, y[w].max() - k * 1.4 - 0.1, f"{q - r:+d} ms", fontsize=7.5, ha="right", color=C_P)
    ax.set_xlabel("time from Pan–Tompkins mark (ms)"); ax.set_yticks([]); fs.despine(ax, left=True)
    ax.set_title("two beats", fontsize=8.5)

    # C: lag histogram
    ax = fig.add_subplot(gs[1, 1]); fs.panel(ax, "C")
    ax.hist(np.clip(lag, -20, 60), bins=np.arange(-20, 62, 2), color=C_P, alpha=0.85)
    ax.set_xlabel("Pan–Tompkins − Brammer (ms)"); ax.set_ylabel("beats")
    ax.set_title(f"{100*late.mean():.0f}% of marks 15–60 ms late", fontsize=8.5)

    # D: RR series
    ax = fig.add_subplot(gs[1, 2]); fs.panel(ax, "D")
    n0 = np.searchsorted(b, s0); rb = np.diff(b)[n0:n0 + 60]
    m0 = np.searchsorted(p, s0); rp = np.diff(p)[m0:m0 + 60]
    ax.plot(np.arange(len(rp)), rp, color=C_P, lw=1.0, label="Pan–Tompkins")
    ax.plot(np.arange(len(rb)), rb, color=C_B, lw=1.3, label="Brammer")
    ax.set_xlabel("beat"); ax.set_ylabel("RR interval (ms)")
    ax.set_title("same recording", fontsize=8.5)

    # E: group curves
    ax = fig.add_subplot(gs[2, :2]); fs.panel(ax, "E", dx=-0.1)
    cur = pd.read_csv(os.path.join(ROOT, "data", "cetram_detector_curves.csv"))
    tau = np.arange(1, 6)
    for det, ls, lab in [("brammer", "-", "Brammer"), ("pantompkins", "--", "Pan–Tompkins")]:
        for g in ["Control", "PD"]:
            d = cur[(cur.det == det) & (cur.Group == g)][[f"s{i}" for i in tau]]
            mu, se = d.mean().values, d.std().values / np.sqrt(len(d))
            ax.plot(tau, mu, ls=ls, color=fs.GROUP[g], marker="o", ms=3.5, label=f"{g}, {lab} (n={len(d)})")
            ax.fill_between(tau, mu - se, mu + se, color=fs.GROUP[g], alpha=0.12, lw=0)
    ax.set_xticks(tau); ax.set_xlabel(r"scale $\tau$"); ax.set_ylabel("sample entropy")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07)
    print(" ->", fs.save(fig, OUT, "Fig6_detector"))


if __name__ == "__main__":
    main()

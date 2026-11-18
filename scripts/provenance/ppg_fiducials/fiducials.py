"""Step 1, label-blind. Re-detect each Cruces pulse at five fiducial points on the raw 500 Hz PPG.
Beats are located exactly as in prepare_ppg_data.py (nk.ppg_clean + Elgendi systolic peaks), then
for every beat the foot is the minimum of the cleaned PPG in the 400 ms before the apex, and:
  apex     systolic peak, parabolic sub-sample refinement           (current pipeline, refined)
  maxd1    maximum of the first derivative on the upstroke, parabolic refinement
  mid      50% amplitude between foot and apex, linear interpolation
  tangent  intersection of the tangent at maxd1 with the horizontal through the foot (Peralta 2019)
  foot     minimum, parabolic refinement
No group labels are read anywhere in this script."""
import os, glob, numpy as np, pandas as pd, neurokit2 as nk
W = os.path.dirname(os.path.abspath(__file__)); FS = 500.0
keep = set(open('/mnt/user-data/uploads/Multicenter/public_release/scratch/bundle3/analysed_subjects.txt').read().split())
def parab(y, i):
    if i <= 0 or i >= len(y)-1: return float(i)
    a, b, c = y[i-1], y[i], y[i+1]; den = a - 2*b + c
    return float(i) if den == 0 else i + 0.5*(a - c)/den
rows, beats = [], {}
for f in sorted(glob.glob(f'{W}/ppg_traces/*.csv')):
    s = os.path.basename(f)[:-4]
    if s not in keep: continue
    x = np.loadtxt(f); c = nk.ppg_clean(x, sampling_rate=500)
    pk = np.asarray(nk.ppg_findpeaks(c, sampling_rate=500, method='elgendi')['PPG_Peaks'])
    d1 = np.gradient(c); T = {k: [] for k in ('apex', 'maxd1', 'mid', 'tangent', 'foot')}
    for i, p in enumerate(pk):
        lo = max(int(p - 0.4*FS), int(pk[i-1]) + 1 if i else 0)
        if p - lo < 10: [T[k].append(np.nan) for k in T]; continue
        ft = lo + int(np.argmin(c[lo:p+1]))
        if p - ft < 5: [T[k].append(np.nan) for k in T]; continue
        md = ft + int(np.argmax(d1[ft:p+1]))
        T['apex'].append(parab(c, p)); T['foot'].append(parab(-c, ft)); T['maxd1'].append(parab(d1, md))
        half = 0.5*(c[ft] + c[p]); j = ft + int(np.argmax(c[ft:p+1] >= half))
        T['mid'].append(j - 1 + (half - c[j-1])/(c[j] - c[j-1]) if j > ft and c[j] != c[j-1] else float(j))
        T['tangent'].append(md - (c[md] - c[ft])/d1[md] if d1[md] > 0 else np.nan)
    beats[s] = {k: np.array(v)/FS*1000 for k, v in T.items()}   # ms
    beats[s]['apex_raw'] = pk/FS*1000
np.save(f'{W}/beats.npy', beats, allow_pickle=True)
print(f"subjects {len(beats)}")

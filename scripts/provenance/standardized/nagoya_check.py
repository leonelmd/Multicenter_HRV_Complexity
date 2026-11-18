import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def hf(x):
    y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y)//2)); return p[f >= 0.35].sum()/p[f > 0.02].sum()
meta = pd.read_csv(f'{S}/dur/data/metadata/metadata.csv')
st = {r.Subject_ID: sum(int(v)*m for v, m in zip(str(r.Start_Time).split(':'), (3600, 60, 1))) for r in meta.itertuples()}
ser = np.load(f'{S}/std/series.npy', allow_pickle=True).item(); R = pd.read_csv(f'{S}/std/screening.csv')
rows = []
for f in sorted(glob.glob(f'{S}/dur/data/processed_rri/*_RRi.txt')):
    s = os.path.basename(f)[:-8]
    if ('Nagoya', s) not in ser: continue
    d = np.loadtxt(f); t, rr = d[:, 0], d[:, 1]*1000; clk = ((t + st[s]) % 86400)/3600
    raw = rr[(clk >= 18) & (clk < 22)]; raw = raw[(raw >= 300) & (raw <= 2000)]
    # 15-min pieces of the raw 18-22 h series: how many are 'white'?
    piece = [hf(raw[i:i+1000]) for i in range(0, len(raw)-1000, 1000)]
    x = ser[('Nagoya', s)]
    rows.append(dict(s=s, hf_raw=hf(raw), hf_LT=hf(x), frac_pieces_white=np.mean(np.array(piece) >= 0.30),
                     q_hf_piece=np.median(piece), d1=np.mean(np.abs(np.diff(raw)) > 0.2*np.median(raw))*100))
N = pd.DataFrame(rows)
print("Nagoya 18-22 h: median alternating share raw %.3f, after LT %.3f; median share of 15-min pieces %.3f" % (N.hf_raw.median(), N.hf_LT.median(), N.q_hf_piece.median()))
print(N.sort_values('hf_LT', ascending=False).head(10).round(3).to_string(index=False))

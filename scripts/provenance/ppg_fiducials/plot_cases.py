import numpy as np, neurokit2 as nk, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
W = os.path.dirname(os.path.abspath(__file__)); beats = np.load(f'{W}/beats.npy', allow_pickle=True).item()
cases = ['E14', 'C07', 'C19', 'E35', 'E10']
fig, ax = plt.subplots(len(cases), 2, figsize=(15, 2.3*len(cases)), gridspec_kw={'width_ratios': [3, 1.2]})
for r, s in enumerate(cases):
    c = nk.ppg_clean(np.loadtxt(f'{W}/ppg_traces/{s}.csv'), sampling_rate=500); t = np.arange(len(c))/500
    a = 120; sel = (t >= a) & (t < a + 15)
    ax[r, 0].plot(t[sel], c[sel], lw=0.8, color='0.3')
    for k, col in (('apex', 'tab:red'), ('maxd1', 'tab:blue')):
        bt = b = beats[s][k]/1000; m = (bt >= a) & (bt < a + 15)
        ax[r, 0].plot(bt[m], np.interp(bt[m], t, c), 'o', ms=3, color=col, label=k)
    ax[r, 0].set_ylabel(s); ax[r, 0].legend(loc='upper right', fontsize=7)
    x = np.diff(beats[s]['maxd1']); x = x[np.isfinite(x)]
    ax[r, 1].plot(x[:-1], x[1:], '.', ms=2, color='0.2'); ax[r, 1].set_title('Poincaré (maxd1 IBI, ms)', fontsize=8)
plt.tight_layout(); plt.savefig(f'{W}/cases.png', dpi=80)

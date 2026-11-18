import sys; sys.path.insert(0,'ppg')
import numpy as np, pandas as pd, glob, os
from multiprocessing import Pool
from ppg_lib import curve, nauc
from scipy.stats import mannwhitneyu
def job(a):
    det,g,f=a; x=np.diff(pd.read_csv(f)['sample'].values.astype(float))
    c=curve(x); return dict(det=det,Group=g,Subject=os.path.basename(f)[:4],idx=nauc(c),**{f's{i+1}':c[i] for i in range(5)})
jobs=[(d,g,f) for d in ['brammer','pantompkins'] for g in ['Control','PD'] for f in sorted(glob.glob(f'cet/results_replay/{d}/{g}/*_cleaned.csv'))]
with Pool(8) as p: r=pd.DataFrame(p.map(job,jobs))
r.to_csv('ms/pt_vs_brammer.csv',index=False)
for d,g in r.groupby('det'):
    C=g[g.Group=='Control'];P=g[g.Group=='PD'];u=mannwhitneyu(C.idx,P.idx)
    print(d,len(C),len(P),'AUC %.3f p %.4f'%(u.statistic/len(C)/len(P),u.pvalue),
      '| C s1 %.2f s5 %.2f | PD s1 %.2f s5 %.2f'%(C.s1.mean(),C.s5.mean(),P.s1.mean(),P.s5.mean()))

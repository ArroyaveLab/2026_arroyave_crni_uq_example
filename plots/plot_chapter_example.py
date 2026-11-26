"""Plot archived ensemble statistics without recomputing thermodynamics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT/'chapter/data/chapter_numbers.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.6,3.3),layout='constrained')
keys=['joint','indep','laplace']; labels=['Joint sample','Independent\nmarginals','Modified local\nGaussian']
v=[D['decision_state'][k]['P_fcc'] for k in keys]
y=[x['p'] for x in v]; err=[max(x['se_binomial'],x.get('se_cluster_walker',0)) for x in v]
ax.bar(range(3),y,yerr=err,capsize=4,color=['#286089','#b86822','#777777'],width=.6)
ax.set(xticks=range(3),xticklabels=labels,ylim=(0,1.06),ylabel='FCC ensemble frequency')
for i,p in enumerate(y): ax.text(i,p+.045,f'{p:.3f}',ha='center')
ax.set_title(r'$x_{\mathrm{Ni}}=0.52$, $T=1473$ K',fontsize=11)
ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'chapter/figures/fig_24_example_ensemble_comparison.pdf');fig.savefig(ROOT/'chapter/figures/fig_24_example_ensemble_comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(5.6,3.5),layout='constrained')
for kind,label,color,marker in [('x_fcc_boundary','FCC end','#286089','o'),('x_bcc_boundary','BCC end','#b86822','s')]:
 rows=sorted([r for r in D['ranking'] if r['kind']==kind],key=lambda r:r['T'])
 vals=[r['by_sigma']['calibrated_unknown'] for r in rows]
 ax.errorbar([r['T'] for r in rows],[100*v['dU_map_rel'] for v in vals],yerr=[100*v['dU_map_se_boot']/v['U_map_now'] for v in vals],label=label,color=color,marker=marker,capsize=3)
ax.set(xlabel='Measurement temperature (K)',ylabel='Expected reduction of map variance (%)',ylim=(0,50))
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
fig.savefig(ROOT/'chapter/figures/fig_25_example_measurement_value.pdf');fig.savefig(ROOT/'chapter/figures/fig_25_example_measurement_value.png',dpi=180)
print('Created two statistical figures from archived JSON.')

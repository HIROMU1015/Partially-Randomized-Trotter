"""Plots saved scalar diagnostics; never imports scientific execution code."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
x=json.loads((HERE/'coefficient_order_audit.json').read_text());rows=x['records']
q=[v['q'] for v in rows];e=[float(v['operator_frobenius_error']) for v in rows]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].loglog(q,e,'o-',label='stored 8-decimal coefficients')
axes[0].loglog(q,[e[0]/t**4 for t in q],'--',label='q^-4 guide')
axes[0].set(xlabel='number of outer steps q',ylabel='operator Frobenius error',title='70-digit noncommuting toy diagnostic')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
orders=[v['observed_order'] for v in rows[1:]]
axes[1].semilogx(q[1:],orders,'o-');axes[1].axhline(4,color='gray',linestyle='--');axes[1].axhline(2,color='gray',linestyle=':')
axes[1].set(xlabel='q',ylabel='observed order',ylim=(min(-.5,min(orders)-.2),max(orders)+.3),title='Fourth-order range, then coefficient defect')
axes[1].grid(alpha=.25)
fig.savefig(HERE/'coefficient_order.png',dpi=150);fig.savefig(HERE/'coefficient_order.svg');plt.close(fig)
summary=json.loads((HERE/'representative_cost_summary.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for system,ax in zip(('H4','H6'),axes):
 own=[v for v in summary if v['system']==system]
 for i,(v,color) in enumerate(zip(own,['#4477aa','#eeaa33','#228833'])):
  ax.errorbar(i,v['shot_weighted_RZ_mean'],yerr=v['shot_weighted_RZ_SE'],fmt='o',markersize=8,capsize=4,color=color)
 ax.set_xlim(-.5,2.5)
 ax.set_xticks(range(3),['S2 q4','Yoshida q2','new_2 q1'])
 ax.set_yscale('log');ax.set_title(system+' representative PR split')
 ax.set_ylabel('Shot-weighted RZ (P=0), mean ± SE; log scale')
 ax.grid(axis='y',alpha=.25)
fig.suptitle('Signal epsilon=.001, T=.8, n=8; fixed points, not optimized q')
fig.savefig(HERE/'representative_cost.png',dpi=150);fig.savefig(HERE/'representative_cost.svg');plt.close(fig)

# Normalize generated SVG formatting without changing path geometry.
for path in HERE.glob('*.svg'):
 path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

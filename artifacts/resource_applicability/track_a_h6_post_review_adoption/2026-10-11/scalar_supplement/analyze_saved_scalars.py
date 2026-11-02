"""Read-only arithmetic on values transcribed from fixed GitHub evidence.
No repository modules, molecular inputs, solvers, circuits, or samplers are used.
This is a post-hoc review calculation, not an original pilot result or certificate.
"""
import json, math
from pathlib import Path
P=Path(__file__).parent
RESULT='eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2'
# b_re/b_im are the saved signed signal differences, not errors of the oracle check.
rows=[
 dict(id='B0_S2_q2',q=2,order=2,prefix=10,b_re=-0.013559096763192335,b_im=-0.022668570164607216,b_abs=.02641426088955232,logB=0.,rz=[39847],cx=[18188],rz_depth=[5567],cx_depth=[7846],depth=[16009],wrappers=[2,3],wall=18.202693002298474),
 dict(id='B1_S2_q1',q=1,order=2,prefix=19,b_re=.006224107885544394,b_im=.009411797247369402,b_abs=.01128368053413806,logB=0.,rz=[37561],cx=[16960],rz_depth=[5251],cx_depth=[7292],depth=[15041],wrappers=[6,7],wall=14.985598642844707),
 dict(id='B1_S2_q2',q=2,order=2,prefix=19,b_re=.0014336041541876954,b_im=.0022852931061523307,b_abs=.002697737098371816,logB=0.,rz=[74297],cx=[33660],rz_depth=[10387],cx_depth=[14496],depth=[29817],wrappers=[10,11],wall=30.43327384116128),
 dict(id='B1_S4_q1',q=1,order=4,prefix=19,b_re=-.0017969704244354956,b_im=-.0029286250613558273,b_abs=.003435978384142908,logB=0.,rz=[111107],cx=[50480],rz_depth=[15569],cx_depth=[21798],depth=[44726],wrappers=[14,15],wall=44.5960856708698),
 dict(id='B1_S4_q2',q=2,order=4,prefix=19,b_re=-.0001242641701407754,b_im=-.00020150814067998457,b_abs=.00023674271845419034,logB=0.,rz=[221341],cx=[100636],rz_depth=[31015],cx_depth=[43472],depth=[89149],wrappers=[18,19],wall=89.09559024637565),
 dict(id='B2_K2_q2_R4',q=2,order=2,prefix=10,b_re=.0014336044769870382,b_im=.0022852929752350537,b_abs=.0026977371590086317,logB=.00020571974130701606,rz=[43457,43301],cx=[19204,19156],rz_depth=[6002,5990],cx_depth=[8127,8117],depth=[16982,16958],wrappers=[22,23,26,27],wall=48.84100071201101,finite=.00000000034390558561008384,outer=.0026977371558046417),
 dict(id='B3_K6_q2_R4',q=2,order=2,prefix=0,b_re=.0011742948787865082,b_im=.001869925835108921,b_abs=.0022080740683120244,logB=20.213362629147213,rz=[13678],cx=[3856],rz_depth=[1694],cx_depth=[934],depth=[3754],wrappers=[30,31],wall=6.626158538740128,finite=3.2037129064451155e-9,outer=.0022080748655766216),
]
for r in rows:
 r['B']=math.exp(r['logB']);r['B2']=math.exp(2*r['logB'])
 r['axis_nominal_threshold']=math.sqrt(2)*max(abs(r['b_re']),abs(r['b_im']))
 for k in ('rz','cx','rz_depth','cx_depth','depth'):r[k+'_sample_mean']=sum(r[k])/len(r[k])
 r['n_cost_trajectories']=len(r['rz'])
 assert math.isclose(math.hypot(r['b_re'],r['b_im']),r['b_abs'],rel_tol=2e-12)
# J omits the common 2*log(2/alpha) and ceiling from the prior symmetric-axis
# Hoeffding-style policy. u=0 is an explicitly hypothetical nominal sensitivity,
# not an accepted numerical allowance; random C is only a saved sample proxy.
def score(r,eps,u=0.):
 hs=[eps/math.sqrt(2)-abs(r[a])-u for a in ('b_re','b_im')]
 if min(hs)<=0:return None
 return r['B2']*r['rz_sample_mean']*sum(1/h**2 for h in hs)
scenarios=[]
for eps in (.05,.01,.005,.001):
 raw={r['id']:score(r,eps) for r in rows}
 best=min(v for v in raw.values() if v is not None)
 scenarios.append(dict(epsilon=eps,u=0.,scores=raw,ratios_to_smallest_saved_proxy={k:None if v is None else v/best for k,v in raw.items()}))
r={x['id']:x for x in rows}
# Finite-window apparent orders only; not asymptotic certification.
rates={f'S{k}':math.log(r[f'B1_S{k}_q1']['b_abs']/r[f'B1_S{k}_q2']['b_abs'],2) for k in (2,4)}
controls=[dict(id='B1_S4_q2',ordinary=479441,symmetric=221341),dict(id='B2_rep0',ordinary=89127,symmetric=43457),dict(id='B2_rep1',ordinary=88971,symmetric=43301),dict(id='B3_rep0',ordinary=13728,symmetric=13678)]
for c in controls:c['rz_reduction']=1-c['symmetric']/c['ordinary']
key={
 'B2_to_B1S2q2_cost_ratio':r['B2_K2_q2_R4']['rz_sample_mean']/r['B1_S2_q2']['rz_sample_mean'],
 'B2_to_B1S2q2_cost_reduction':1-r['B2_K2_q2_R4']['rz_sample_mean']/r['B1_S2_q2']['rz_sample_mean'],
 'B2_corrected_signal_vs_B1S2q2':math.hypot(r['B2_K2_q2_R4']['b_re']-r['B1_S2_q2']['b_re'],r['B2_K2_q2_R4']['b_im']-r['B1_S2_q2']['b_im']),
 'B2_finite_to_outer':r['B2_K2_q2_R4']['finite']/r['B2_K2_q2_R4']['outer'],
 'B3_finite_to_outer':r['B3_K6_q2_R4']['finite']/r['B3_K6_q2_R4']['outer'],
 'B1S4q2_to_B2_cost_ratio':r['B1_S4_q2']['rz_sample_mean']/r['B2_K2_q2_R4']['rz_sample_mean'],
 'correctness_sum_cell_seconds':sum(x['wall'] for x in rows),
 'saved_peak_RSS_GiB':7092899840/(2**30),
 'saved_peak_RSS_GB':7092899840/1e9,
 'AS_cap_is_not_free_RSS':True,
}
# Local stationary point in a toy single-axis, B=1, C=c*q^a,
# b=A*q^-p model: b/epsilon_axis=a/(a+2p). Do not use to generate truth values.
out=dict(source_commit=RESULT,input_kind='manual transcription of fetched JSON fields, not byte-exact source files',new_molecular_or_circuit_computations=0,posthoc_arithmetic_only=True,rows=rows,nominal_scenarios=scenarios,finite_window_apparent_orders=rates,control_examples=controls,key_results=key,limitations=['No raw state vectors independently re-evolved','No population mean from n=1 or n=2','No numerical allowance certified','No original N/G null fields changed','Conditional proxy scores are not a method winner or accepted budget'])
(P/'saved_scalar_review_calculations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(key,indent=2));print('Apparent orders',rates)
for x in rows:print(x['id'],f'B={x["B"]:.10g} B2={x["B2"]:.10g} eps_min={x["axis_nominal_threshold"]:.10g} C={x["rz_sample_mean"]}')
for x in scenarios:print('EPS',x['epsilon'],x['ratios_to_smallest_saved_proxy'])

"""Independent checks on saved scalar evidence, budget enumeration and source bytes.

No trajectory generation, scientific arrays, circuit build or scientific imports.
This is a local metadata audit, not a scientific certificate or CI run.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RA=ROOT/'artifacts/resource_applicability'
checks=[]

def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

def read(p): return json.loads(p.read_text())
def close(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-6)
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))

receipt=read(HERE/'input_metadata_receipt.json')
for a in receipt['inputs']:
    p=ROOT/a['path']; check('input bytes and SHA: '+a['path'],
       p.stat().st_size==a['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'])
closure=read(RA/'track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/source_closure_v1.json')
check('all old runtime source bytes match fixed pilot closure',all(
    hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256'] for a in closure['files']))
check('257 inherited plus pilot runtime files preserved',len(closure['files'])==257)
conf=read(RA/'track_a_h6_confirmation_summary_v1/2026-10-12/result_summary_v1.json')
exp=read(RA/'track_a_h6_exploration_summary_v1/2026-10-12/result_summary_v1.json')
metrics=csvrows(RA/'track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv')
panel=csvrows(HERE/'h6_fixed_panel.csv')
selection={p['cell_id'] for a in exp['comparisons'] if a['allowance_factor']=='1.0'
           for p in a['family_top2_point']['B2']}
check('eight configurations equal original pre-result top2 union',selection=={x['cell_id'] for x in panel} and len(selection)==8)
for a in conf['comparisons']:
    if a['allowance_factor']!='1.0': continue
    for p in a['B2_top2']:
        m=[x for x in metrics if x['cell_id']==p['cell_id']]
        pairs={}
        for x in m:pairs.setdefault(int(x['replica']),{})[x['axis']]=x
        check('32 paired independent saved samples '+p['cell_id'],len(pairs)==32 and all(set(x)=={'cosine','sine'} for x in pairs.values()))
        g=[sum(p['axes'][axis]['shots']*int(x[axis]['RZ']) for axis in ('cosine','sine')) for x in pairs.values()]
        check('independent CSV G and paired SE '+p['cell_id'],close(statistics.mean(g),p['G_point']) and close(statistics.stdev(g)/math.sqrt(32),p['G_standard_error']))
        display=next(x for x in panel if x['cell_id']==p['cell_id'])
        check('display G/PF and saved ratio '+p['cell_id'],close(float(display['G']),statistics.mean(g)) and close(float(display['ratio']),statistics.mean(g)/a['family_best_point']['B1']['G_point']))
    p2,p4=sorted(a['B2_top2'],key=lambda p:p['cell_id'])
    check('K2/K4 difference within descriptive two-SE '+str(a['epsilon']),abs(p2['G_point']-p4['G_point'])<=2*math.hypot(p2['G_standard_error'],p4['G_standard_error']))
pilot=read(RA/'track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/cost_contribution_analysis_v1.json')
pm=csvrows(RA/'track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/pilot_wrapper_metrics_v1.csv')
check('pilot has 16 saved paired wrappers',len(pm)==16)
mu2=statistics.mean(int(x['RZ']) for x in pm)
check('pilot observed RZ mean from CSV',close(mu2,96285.25))
last=pilot['comparisons'][-1]; n=sum(last['saved_shot_counts'].values());mu0=last['s0_confirmation']['mean']/n
new=(1-pilot['s2_probability'])*mu0+pilot['s2_probability']*mu2
check('pilot known-weight mean update and G',close(new,95870.818474227) and close(new*n,last['stratified_G_point_estimate']))
plan=read(HERE/'h8_plan_draft.json')
check('no technical grant or runnable command',not plan['approved'] and not plan['runtime_authorization'] and plan['grant'] is None and plan['allowed_cpus']==[] and plan['absolute_launch_command'] is None and not plan['technical_sealed'])
check('H8 actual input and rank remain unknown',plan['target_proposal']['actual_DF_rank'] is None and plan['target_proposal']['Hamiltonian_SHA'] is None and not plan['memory_arithmetic']['H8_peak_memory_measured'])
for key,qs in [('core_budget_upper',(1,2,4,8)),('q16_option_budget_upper',(1,2,4,8,16))]:
    b=plan[key]
    deterministic=list(itertools.product(('B0_p1','B0_p2','B0_p3','B1_S2','B1_S4'),qs))
    random=list(itertools.product((1,2,3),qs,(1,2,4),(2,4),range(8)))
    check(key+' independent enumeration',b['signal_candidates']==5*len(qs)+3*len(qs)*6 and b['exploration_groups_upper']==len(deterministic)+len(random))
    check(key+' paired compile/trajectory and worst occurrences',b['new_compile_upper']==2*(len(deterministic)+len(random)+256+40)+4 and b['new_trajectories_upper']==len(random)+296 and b['random_occurrences_upper']==sum(x[1] for x in random)+296*max(qs))
check('carry preserves reservations no refund',plan['projected_compile_total_upper_core']==3524 and plan['projected_trajectory_total_upper_core']==1724 and not plan['prior_carry']['reset'] and plan['prior_carry']['refund']==0)
spec=importlib.util.spec_from_file_location('metadata_builder',HERE/'build_evidence.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
check('prefix scenarios and small rank dedup',mod.prefixes(19)==[5,10,15] and mod.prefixes(2)==[1] and mod.prefixes(24)==[6,12,18])
check('weighted 8 normal admitted; heavy combinations rejected',mod.permitted_mix(0,8) and mod.permitted_mix(1,5) and not mod.permitted_mix(1,6) and not mod.permitted_mix(8,0) and not mod.permitted_mix(-1,1))
check('dense-sector dimension and memory arithmetic',plan['memory_arithmetic']['sector_dimension']==4900 and plan['memory_arithmetic']['one_dense_sector_matrix_bytes']==384160000 and plan['memory_arithmetic']['dense_full_system_unitary_bytes']==64*2**30)
docs=[ROOT/'docs/manuscripts/track_a_h6_resource_study_v0_2.md',ROOT/'docs/research/track_a_h4_h6_common_comparison_plan_v1.md',ROOT/'docs/research/track_a_h8_resource_comparison_plan_v1.md',HERE/'README.md']
broken=[]
for d in docs:
    for target in re.findall(r'\]\(([^)]+)\)',d.read_text()):
        if target.startswith(('http:','https:','#')):continue
        if not (d.parent/target.split('#')[0]).exists():broken.append((str(d.relative_to(ROOT)),target))
check('new manuscript and plan local links exist',not broken)
check('three SVGs contain expected exported plots',all((HERE/('FigH6_'+n)).read_text().count('<svg')==1 for n in ['1_precision.svg','2_cost_factors.svg','3_pilot.svg']))
result={'status':'LOCAL_METADATA_AND_SCALAR_CHECKS_PASS','checks_passed':len(checks),'checks':checks,
 'old_runtime_source_files_verified':len(closure['files']),'new_scientific_computation':0,
 'H8_scientific_validation':'NOT_RUN','H8_performance':'UNMEASURED','CI_run':False,
 'population_mean_bound':None,'new_grant':None}
(HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))

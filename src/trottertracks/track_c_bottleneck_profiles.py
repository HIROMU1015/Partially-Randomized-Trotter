"""Additive basis/diagonal/control attribution; exposed cuts are subsets.

The chosen modular executable determines the total. Isolated template
attribution has an explicit signed compiler/context residual. In particular
depth residuals include scheduling and are not gate-cancellation savings.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import subprocess
import numpy as np
from .track_c_bottleneck import OUT, ROOT, PRIOR, NAMES, saved, components
from .track_c_bottleneck_select import replacement_prices, new_point
from .track_c_bottleneck_runwise import d_profile, build_singletons
from .track_c_bottleneck_basis import profile_rte
from .track_c_h4 import save_json
from .track_c_d_resources import r_key

PARTS = ('basis','uncontrolled_diagonal','control_increment','relative_phase','compile_correction')


def cell_profile(task):
    key, strata = task
    means = {arm:{k:np.zeros(4) for k in (*PARTS,'boundary_exposed_basis')} for arm in ('old','selected')}
    for records in strata.values():
        probability = records[0]['probability']
        for record in records:
            if 'old_profile' in record:
                old, proposed = record['old_profile'], record['proposal_profile']
            else:
                old = profile_rte({**record,'metrics':record['old']})
                _, proposed = build_singletons(record['t'],record['n'],record['choices'])
                proposed['compile_correction'] = (np.array(record['proposal'])-sum((np.array(proposed[k]) for k in PARTS[:-1]),np.zeros(4))).tolist()
            chosen = proposed if record['choice']=='singletons' else old
            for arm, profile in (('old',old),('selected',chosen)):
                total = sum((np.array(profile[k]) for k in PARTS),np.zeros(4))
                assert np.allclose(total,record[arm],rtol=1e-13,atol=1e-7)
                for k in means[arm]: means[arm][k] += probability*np.array(profile[k])/len(records)
    return key,{arm:{k:v.tolist() for k,v in profile.items()} for arm,profile in means.items()}


def run():
    if (OUT/'TEMPLATE_COMPONENTS.json').exists(): raise RuntimeError('Do not overwrite profiles')
    records = replacement_prices(); cert=json.loads((OUT/'REPLACEMENT_CERTIFICATE.json').read_text())
    qs=dict(saved()[1])
    for q in json.loads((OUT/'JOINT_NATIVE_SOURCE.json').read_text())['rows']: qs[(q['name'],q['m'],q['h'],q['rate'])]=q
    from .track_c_bottleneck import identity
    selected=[]
    for name in NAMES:
        pp=[components(r) for r in sorted(saved()[0][name]['selected'],key=lambda r:r['signal']['m'])]
        selected.append((name,'old',pp,[[p['qualification']['shots_per_axis']]*2 for p in pp]))
    for policy in json.loads((OUT/'OPTIMIZATION.json').read_text())['optimized']:
        if policy['primary']!='RZ': continue
        pp=[new_point(s['signal'],qs[identity(s['signal'])],records,cert) for s in policy['selected']]
        selected.append((policy['family']+'_joint_RZ','selected',pp,[r['shots'] for r in policy['rounds']]))
    needed={r_key(r['time'],r['r']) for _,_,pp,_ in selected for p in pp for r in p['signal']['recipe']}
    profiles={}
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(cell_profile,(key,records[key])) for key in needed):
            key,value=f.result();profiles[key]=value
    result=[];flat=[]
    for name,arm,points,shots in selected:
        rows=[]
        for p,nn in zip(points,shots):
            x=p['signal'];nshots=sum(nn);d=d_profile({'signal':x});d.setdefault('relative_phase',[0]*4)
            r={k:np.zeros(4) for k in (*PARTS,'boundary_exposed_basis')}
            for recipe in x['recipe']:
                profile=profiles[r_key(recipe['time'],recipe['r'])][arm]
                for k in r:r[k]+=x['steps']*np.array(profile[k])
            dd={k:nshots*np.array(d[k]) for k in PARTS};rr={k:nshots*r[k] for k in PARTS}
            wrapper=np.sum(np.array(nn)[:,None]*p['overhead'],axis=0)
            total=sum(dd.values())+sum(rr.values())+wrapper
            assert np.allclose(total,np.sum(np.array(nn)[:,None]*p['mean'],axis=0),rtol=1e-13,atol=1e-5)
            exposed=nshots*(np.array(d['D_R_exposed_basis'])+r['boundary_exposed_basis'])
            row=dict(m=x['m'],D={k:v.tolist() for k,v in dd.items()},RTE={k:v.tolist() for k,v in rr.items()},wrapper=wrapper.tolist(),total=total.tolist(),D_R_exposed_basis_subset=exposed.tolist())
            rows.append(row)
            for group,parts in (('D',dd),('RTE',rr)):
                for component,value in parts.items():
                    for i,metric in enumerate(('RZ','CX','serial_depth_upper','size')):flat.append(dict(policy=name,m=x['m'],group=group,component=component,metric=metric,value=value[i]))
        totals={group:{k:np.sum([r[group][k] for r in rows],axis=0).tolist() for k in PARTS} for group in ('D','RTE')}
        for k in ('wrapper','total','D_R_exposed_basis_subset'):totals[k]=np.sum([r[k] for r in rows],axis=0).tolist()
        result.append(dict(name=name,rows=rows,totals=totals,ideal_zero_exposed_basis_G=(np.array(totals['total'])-totals['D_R_exposed_basis_subset']).tolist()))
    save_json(OUT/'TEMPLATE_COMPONENTS.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),policies=result,
        convention='D/R exposed basis is a subset of basis, never an extra charge. Removing all exposed gates is a deliberately unrealizable fixed-shot sensitivity, not a lower bound or new implementation. Basis+uncontrolled diagonal+incremental control+relative phase+signed context residual equals each actual price. Wrapper contains deterministic scalar phase and X/Y Hadamard/readout. Depth is serial bound; context residual includes DAG scheduling.',
        statistical='Training MC attribution; fixed-policy independent holdout has its own totals/covariance. Exact D and wrapper counts. No decomposition is treated as an independent MC draw.'))
    with (OUT/'TEMPLATE_COMPONENTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    print('TEMPLATE_COMPONENTS_COMPLETE',flush=True)


if __name__=='__main__':run()

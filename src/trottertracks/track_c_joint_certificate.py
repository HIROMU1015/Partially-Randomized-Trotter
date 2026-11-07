"""Directed source/Fock and all-context native bounds for the frozen unions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
import mpmath as mp
from .track_c_joint_isometry import OUT,PRIOR,source,decode,configuration
from .track_c_bottleneck_certificate import one_particle,ops_key
from . import track_c_compiler_a_18round as ca
from . import track_c_source_guarantee as sg
from .track_c_h4 import common,save_json


def column_bound(entry):
    mp.iv.prec=192;original=common()['preps'][3].rte_preparation.basis_registry.definition(entry['basis'])
    a=one_particle(ops_key(original.runtime_operations));b=one_particle(ops_key(decode(entry['operations'])));delta=[]
    for target,old in zip(entry['targets'],entry['support']):
        delta.append(sg.up(mp.iv.sqrt(sum((abs(a[i][old]-b[i][target])**2 for i in range(8)),sg.I(0)))))
    error=sg.up(16*sum((sg.I(x) for x in sorted(delta,reverse=True)[:2]),sg.I(0)))
    return dict(key=entry['key'],column_error_uppers=delta,involution_error_upper=error,
                proof='Individual column projectors, not span alone. Ne=4, 4*Ne*delta per Z; at most two Z via telescoping. Adjacent Gaussian source gates conserve number; native errors use full local matrices.')


def run():
    if (OUT/'CERTIFICATE.json').exists():raise RuntimeError('Do not overwrite certificate')
    entries=[r for r in json.loads((OUT/'CATALOGUE_FINAL.json').read_text())['entries'].values() if r['adopt']]
    configuration();catalogue={};seen=set()
    for spec in ca.inventory()['catalogue'].values():
        if spec['family'] not in seen:catalogue[spec['key']]=spec;seen.add(spec['family'])
    for entry in entries:ca.basis_counter(decode(entry['operations']),catalogue)
    unique={};contexts=[];column=[]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(column_bound,r) for r in entries):column.append(f.result())
        for f in as_completed(pool.submit(ca.context_compile,(mask,catalogue)) for mask in range(128)):
            result=f.result();contexts.append({k:result[k] for k in ('mask','context','post_unroll')})
            for record in result['records']:unique[record['lowering_sha256']]=record
        checked=list(pool.map(ca.interval_check,unique.values()))
    bounds=json.loads((PRIOR/'REPLACEMENT_LOCAL_INTERVALS.json').read_text())['primitive_bounds'].copy()
    for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
    native=[]
    for entry in entries:
        counts=ca.basis_counter(decode(entry['operations']),{})
        native.append(sg.add(*[sg.mul(v,bounds[k]) for k,v in counts.items()]))
    old=json.loads((PRIOR/'REPLACEMENT_CERTIFICATE.json').read_text())
    save_json(OUT/'LOCAL_INTERVALS.json',dict(contexts=contexts,records=list(unique.values()),checked=checked,primitive_bounds=bounds))
    save_json(OUT/'CERTIFICATE.json',dict(source_commit=source(),catalogue_source=json.loads((OUT/'CATALOGUE_FINAL.json').read_text())['source_commit'],
        columns=column,max_involution_error=max(old['max_involution_error'],max(r['involution_error_upper'] for r in column)),
        max_native_basis_pair_error=max(old['max_native_basis_pair_error'],max(native)),
        contexts=len(contexts),unique_native_lowerings=len(unique),
        fallback='Maximum includes every legacy singleton basis as well as all selected joint layouts. Unseen unions use full basis. Bound uniform over all trajectories, not just observed draws.',
        scope='Saved H4, four-electron input, ideal native, fresh finite bits. Reuse old unsupplemented logical qualification; new supplement replaces the old one, never charges it twice.'))
    print('JOINT_CERTIFICATE_COMPLETE',len(unique),flush=True)


if __name__=='__main__':run()

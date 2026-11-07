"""Four prescribed QR layouts, not an unbounded permutation search.

Column order (ascending/descending) and edge (leading/trailing) are the only
extra freedoms. No phase pruning, small-angle cutoff, probability or PF change.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import json
import os
import numpy as np
from qiskit.quantum_info import Operator
from .track_c_joint_isometry import OUT,PRIOR,configuration,circuit,partial_ops,decode,encode,basis_price,key,source,fock_check,build_joint
from .track_c_h4 import common,save_json
from .track_c_d_resources import metrics
from .track_c_compiler_a import compile_reference


def alternatives(u,union):
    rows=[]
    for reverse in (False,True):
        support=tuple(reversed(union)) if reverse else tuple(union)
        for reflected in (False,True):
            ops=partial_ops(u[::-1] if reflected else u,support)
            if reflected:ops=tuple((g,tuple(7-i for i in wires)) for g,wires in ops)
            targets=[7-j if reflected else j for j in range(len(support))]
            price=sum((np.array(metrics(compile_reference(circuit(ops,9,inverse)))) for inverse in (True,False)),np.zeros(4))
            rows.append(dict(support=list(support),targets=targets,reverse=reverse,reflected=reflected,
                             operations=encode(ops),joint_basis_pair=price.tolist()))
    return rows


def refine(entry):
    b=configuration()[3][entry['basis']];idx=[1<<i for i in range(8)];u=b[np.ix_(idx,idx)]/b[0,0]
    rows=alternatives(u,tuple(sorted(entry['support'])));best=min(rows,key=lambda r:tuple(r['joint_basis_pair']))
    return {**entry,**best,'adopt':best['joint_basis_pair'][0]<entry['full_basis_pair'][0],
            'layouts':[dict(reverse=r['reverse'],reflected=r['reflected'],metrics=r['joint_basis_pair']) for r in rows]}


def full_probe(index):
    b=common()['preps'][12].deterministic_blocks[index];v=Operator(circuit(b.runtime_basis_operations)).data;idx=[1<<i for i in range(8)];u=v[np.ix_(idx,idx)]/v[0,0]
    old=sum((np.array(metrics(compile_reference(circuit(b.runtime_basis_operations,9,inverse)))) for inverse in (True,False)),np.zeros(4))
    rows=alternatives(u,tuple(range(8)));best=min(rows,key=lambda r:tuple(r['joint_basis_pair']))
    return dict(block=index,full_basis_pair=old.tolist(),best_QR_pair=best['joint_basis_pair'],improves=best['joint_basis_pair'][0]<old[0],
                layouts=[dict(reverse=r['reverse'],reflected=r['reflected'],metrics=r['joint_basis_pair']) for r in rows])


def price(path):
    target=OUT/'training_final'/path.name
    if target.exists():return json.loads(target.read_text())
    r=json.loads(path.read_text());q,used=build_joint(r['t'],r['n'],r['choices']);proposal=metrics(compile_reference(q))
    result={**r,'baseline':r['selected'],'joint_proposal':proposal,'selected':proposal if proposal[0]<r['selected'][0] else r['selected'],
            'choice':'joint' if proposal[0]<r['selected'][0] else 'baseline','used_runs':used}
    save_json(OUT/'training_final'/path.name,result);return result


def run():
    if (OUT/'CATALOGUE_FINAL.json').exists():raise RuntimeError('Do not overwrite final catalogue')
    (OUT/'training_final').mkdir(exist_ok=False);configuration();entries=[];full=[];checks=[]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(refine,r) for r in json.loads((OUT/'CATALOGUE.json').read_text())['entries'].values()):entries.append(f.result())
        for f in as_completed(pool.submit(full_probe,j) for j in range(13)):full.append(f.result())
        save_json(OUT/'CATALOGUE_FINAL.json',dict(source_commit=source(),entries={r['key']:r for r in entries},rule='Four prescribed QR layouts per union. Strict native RZ saving; unseen union full fallback; original full/singleton whole fallback retained.'))
        for f in as_completed(pool.submit(fock_check,r) for r in entries if r['adopt']):checks.append(f.result())
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        prices=list(pool.map(price,sorted((PRIOR/'holdout_prices').glob('*.json'))))
    save_json(OUT/'LAYOUTS.json',dict(source_commit=source(),full=full,Fock_checks=checks,entries=len(entries),adopted=sum(r['adopt'] for r in entries),samples=len(prices)))
    print('LAYOUTS_COMPLETE',sum(r['adopt'] for r in entries),'Full_improved',sum(r['improves'] for r in full),flush=True)


def resume():
    if (OUT/'LAYOUTS_RESUMED.json').exists():raise RuntimeError('Do not overwrite resumed layouts')
    entries=json.loads((OUT/'CATALOGUE_FINAL.json').read_text())['entries'];full=[];checks=[]
    before=sorted(p.name for p in (OUT/'training_final').glob('*.json'));configuration()
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(full_probe,j) for j in range(13)):full.append(f.result())
        for f in as_completed(pool.submit(fock_check,r) for r in entries.values() if r['adopt']):checks.append(f.result())
        prices=list(pool.map(price,sorted((PRIOR/'holdout_prices').glob('*.json'))))
    save_json(OUT/'LAYOUTS_RESUMED.json',dict(source_commit=source(),catalogue_source=json.loads((OUT/'CATALOGUE_FINAL.json').read_text())['source_commit'],
        preserved_training_files=before,full=full,Fock_checks=checks,entries=len(entries),adopted=sum(r['adopt'] for r in entries.values()),samples=len(prices),
        technical_fix='Sort remapped ZZ support to meet existing dataclass contract; ZZ is symmetric. First attempt stopped with 201 completed files, preserved byte-for-byte. No catalogue extension or physical change.'))
    print('LAYOUTS_RESUMED_COMPLETE','Full_improved',sum(r['improves'] for r in full),flush=True)


if __name__=='__main__':
    import sys
    resume() if len(sys.argv)>1 and sys.argv[1]=='resume' else run()

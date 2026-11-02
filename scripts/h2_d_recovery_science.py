"""Grant-only H2 D recovery: no new signal, shot design, reference, or sampling."""
from pathlib import Path
import sys
from types import SimpleNamespace

import h2_d_recovery_contract as c


def context(spec,output,calls,loader=None):
    config=spec['native_config'];sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    import h4_native_science as h
    from h2_tol_only_input import load
    from h4_dp_science import compiler
    m=h.provider();calls.take('saved_input_load')
    ham,sector,full,state,md=(loader or load)(config)
    c.require((ham.n_qubits,len(ham.lambdas),sector.dimension)==(4,3,4)
        and md['hamiltonian_hash']==c.old.BINDING['hamiltonian']
        and md['state_hash']==c.old.BINDING['state'],'SAME_F_STATE_AND_LAYOUT')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    original=c.read(spec['saved_preparations'])['rows'];cache={};preps={}
    for cell in c.cells():
        raw=cell['id'].replace('B0I_','B0_',1);key=(cell['method']!='B1',cell['prefix'])
        if key not in cache:
            calls.take('preparation');cache[key]=(m._prepare_discard if key[0] else m._prepare)(ham,key[1])
        prep=cache[key]
        c.require(prep.preparation_hash==original[raw]['preparation_hash']
            and prep.hamiltonian_hash==original[raw]['hamiltonian_hash'],'UNCHANGED_NATIVE_PREPARATION')
        preps[raw]=prep
    c.require(len(cache)==2,'EXACT_TWO_NATIVE_PREPARATIONS')
    c.save(output/'preparations.json',dict(binding=c.old.BINDING,rows={k:dict(preparation_hash=p.preparation_hash,
        hamiltonian_hash=p.hamiltonian_hash) for k,p in preps.items()},new_signal_reference_solver=0))
    return SimpleNamespace(ham=ham,preps=preps,qstate=qstate),m,compiler(m,config)


def execute(spec,grant,output):
    c.authorize(spec,grant)
    from h2_panels_science import compile_one,batch
    calls=c.old.Calls(output,spec['calls'])
    records=c.read(spec['saved_signals'])['signals'];byid={r['cell']['id']:r for r in records}
    port,m,compiler=context(spec,output,calls)
    tasks=[dict(id=x['id'],cell=x,replica=0,expected=byid[x['id']]) for x in c.cells()]
    calls.take('native_evolution',28);calls.take('wrapper_compile',56);calls.take('control_probe',112)
    rows=batch(spec,tasks,lambda t,d:compile_one(port,m,compiler,t,d,'D'),output/'compile_workers')
    c.require(len(rows)==28 and {r['cell_id'] for r in rows}=={x['id'] for x in c.cells()},'COMPLETE_FRESH_D_COSTS')
    fp=c.read(spec['FP_result']);fpids={x['id'] for x in c.old.registrations()['FP'] if not x['R']}
    reused=[r for r in fp['measurements'] if r['cell_id'] in fpids]
    c.require(len(reused)==6,'SIX_NATIVE_FP_PAIRS')
    pairs={}
    for r in [*reused,*rows]:
        c.old.check_pair(r);c.require(r['cell_id'] not in pairs,'NO_PARTIAL_COST_POOLING')
        pairs[r['cell_id']]=[r]
    calls.take('endpoint_accounting_pass')
    accounting=c.old.account(records,pairs,'D')
    c.save(output/'signals.json',dict(signals=records,binding=c.old.BINDING,reused_signal_records=70,
        original_sha256=c.sha(spec['saved_signals']),new_signal=0))
    return dict(scope=c.SCOPE,stage='D',status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],
        original_SOURCE=c.OLD_SOURCE,binding=c.old.BINDING,signals=records,reference=fp['reference'],
        measurements=rows,reused_FP_measurements=reused,accounting=accounting,
        calls_reserved=calls.used,calls_maximum=spec['calls'],original_D_compile_axes_reserved=56,
        cumulative_D_compile_axes_reserved=56+calls.used['wrapper_compile'],old_reservation_refunds=0,
        old_partial_compile_pairs_reused=0,reused_signal_records=70,new_signed_shot_endpoints=0,
        preparation_identities={k:p.preparation_hash for k,p in port.preps.items()},
        automatic_retry=False,next_stage_authorized=False)

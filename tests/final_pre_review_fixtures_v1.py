"""Fabricated scalar/circuit fixtures. Never reads scientific result files."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


driver = load('fixture_h8_driver', 'h8_d0_saved_execution_v1.py')
bridge = load('fixture_h6_bridge', 'h6_native_b0i_bridge_v1.py')
g, a, d0 = driver.g, driver.a, driver.d0
BUDGET = dict(address_space_bytes=512*2**20, rss_bytes=256*2**20, cpu_seconds=240, wall_seconds=300,
    output_bytes=32*2**20, terminal_reserve_bytes=2**20, output_inode_cap=100, terminal_inode_reserve=4,
    monitor_max_seconds=5., fresh_memory_available_bytes=2**30,
    fresh_available_bytes=256*2**20, fresh_available_inodes=20000)


def authorization(contract, output):
    return dict(scope=contract['scope'], approved_by_user=True, runtime_authorization=True,
        contract_sha256=d0.digest(contract), SOURCE=contract['SOURCE'], automatic_retry=False,
        exclusive_output=str(output), approval_evidence_sha256='0'*64, review_commit='1'*40,
        synthetic_only=True,
        **{k+'_digest': d0.digest(contract[k]) for k in
            ('input_receipts', 'budget', 'binding', 'profile', 'source_hashes', 'allowed_cpus', 'carry')})


def base_contract(scope, root, output, binding):
    root.mkdir(exist_ok=True)
    (root/'synthetic_source.txt').write_text('ARTIFICIAL TEST SOURCE\n')
    return dict(scope=scope, approved=True, runtime_authorization=True, synthetic_only=True,
        SOURCE='a'*40, automatic_retry=False, next_stage_authorized=False, exclusive_output=str(output),
        input_receipts=[], budget=deepcopy(BUDGET), binding=deepcopy(binding),
        profile=dict(python=sys.version, executable=sys.executable, runtime_file_hashes={}),
        source_hashes={'synthetic_source.txt':g.sha((root/'synthetic_source.txt').read_bytes())},
        runtime_root=str(root),
        allowed_cpus=[], carry=dict(actual=0, reset=False, refund=0))


def synthetic_h8():
    cells = a.registry()
    selected = [f'B2_p7_q{q}_2nd_r4_K{K}' for q in (2, 4, 8) for K in (2, 4)]
    binding = dict(analysis_source=a.D0_SOURCE, source='a'*40, seed_source='b'*40,
        result='c'*40, input='d'*64, compiler='e'*64, allowed_wrapper_sources=['a'*40])
    signals, wrappers = [], []
    for index, cell in enumerate(cells):
        bias = .5 if cell['method']=='B0' and cell['prefix']==7 else 0.
        value = 500 if cell['method'] != 'B2' else 1000+index
        if cell['id'] in selected:
            bias, value = {2:(.009,5), 4:(.0005,20), 8:(0.,30)}[cell['q']]
        s = dict(fixture_kind='ARTIFICIAL', cell=cell, b=1., log_B=0.,
            numerical_allowance_certified=False,
            allowance=dict(real=1e-11, imag=1e-11, log_B_margin=1e-11, kind='EMPIRICAL', certified=False),
            error_decomposition=dict(total_signed=dict(real=(.7+bias)-.7, imag=(.2+bias)-.2)),
            reference=dict(real=.7, imag=.2), signals=dict(corrected=dict(real=.7+bias, imag=.2+bias)))
        s['precision_rows'] = {str(e):{str(f):d0.accounting(s,e,f) for f in a.FACTORS} for e in a.ORIGINAL_EPSILONS}
        signals.append(s)
        if bias == .5: continue
        phases = [('exploration', 8 if cell['method']=='B2' else 1)]
        if cell['id'] in selected: phases.append(('confirmation',32))
        for phase, count in phases:
            for i in range(count):
                for axis in ('cosine','sine'):
                    v = value+i*(1 if axis=='cosine' else 2)
                    metrics = {m:v for m in d0.METRICS}
                    if cell['id']=='B2_p7_q8_2nd_r1_K2': metrics['CX']=1
                    wrappers.append(dict(cell_id=cell['id'],phase=phase,replica=i,
                        seed=a.task_seed(binding,cell['id'],phase,i),axis=axis,source=binding['source'],
                        compiler_hash=binding['compiler'],control='symmetric_directional',
                        event_digest=f'ARTIFICIAL/{phase}/{i}',**metrics))
    by = {s['cell']['id']:s for s in signals}
    selection = dict(selected_ids=selected, fixed_before_confirmation=True, exploration_confirmation_pooled=False,
        signal_digest=d0.digest(signals), registrations=[dict(task=dict(cell_id=cid,cell=by[cid]['cell'],
            replica=i,phase='confirmation',seed=a.task_seed(binding,cid,'confirmation',i),
            expected_signal=by[cid]['signals']['corrected'])) for cid in selected for i in range(32)])
    binding['selection_digest']=d0.digest(selection)
    old_maps={};specs=driver.specs(binding,selected)
    for scope,spec in specs.items():
        records=d0.analyze_scope(signals,[r for r in wrappers if r['phase']==spec['primary_phase']],spec,
            a.ORIGINAL_EPSILONS,a.FACTORS,[r for r in wrappers if r['phase']=='exploration'])
        rows=[]
        for r in records:
            if r['status']!='DERIVED_SAVED_SCALAR':continue
            pairs=d0.paired_rows([w for w in wrappers if w['phase']==r['sample_phase']],
                cell_id=r['cell_id'],phase=r['sample_phase'],compiler_hash=binding['compiler'])
            for metric in d0.METRICS:
                import statistics
                axes={}
                for axis,channel in d0.AXIS_CHANNEL.items():
                    values=[p[axis][metric] for p in pairs];sd=statistics.stdev(values) if len(values)>1 else None
                    axes[axis]=dict(n=len(values),mean=statistics.mean(values),min=min(values),max=max(values),
                        sample_sd=sd,standard_error=sd/len(values)**.5 if sd is not None else None,
                        shots=r['accounting']['axes'][channel]['shots'],
                        population_mean_certified=False,interval_is_confidence_bound=False)
                rows.append(dict(cell_id=r['cell_id'],method=r['method'],epsilon=r['epsilon'],
                    allowance_factor=r['factor'],metric=metric,axes=axes,G_point=r['metrics'][metric]['G_point'],
                    G_standard_error=r['metrics'][metric]['G_standard_error'],N_total=r['N_total'],
                    phase=spec['primary_phase'],common_preparation_slope=r['N_total'],preparation_cost_P=0.,
                    formal_winner_certified=False))
        old_maps[scope]=dict(rows=rows)
    import csv,io
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(wrappers[0]));writer.writeheader();writer.writerows(wrappers)
    payloads={'signal_summary.json':json.dumps(dict(signals=signals)).encode(),
        'selection_before_confirmation.json':json.dumps(selection).encode(),
        'wrapper_metrics.csv':stream.getvalue().encode(),
        'completion_audit.json':json.dumps(dict(fixture_kind='ARTIFICIAL',SOURCE=binding['source'],
            seed_source=binding['seed_source'],input_SHA=binding['input'],compiler_hash=binding['compiler'],
            selected_ids=selected,Taylor_orders={'exploration':{'0':1,'4':1},'confirmation':{'0':1}})).encode(),
        'exploration_resource_map.json':json.dumps(old_maps[driver.SCOPES[0]]).encode(),
        'confirmation_resource_map.json':json.dumps(old_maps[driver.SCOPES[1]]).encode()}
    return binding,payloads,sum(len(m['rows']) for m in old_maps.values())


class FakeMonitor:
    def __init__(self,grant):
        self.phase='PRELAUNCH';self.first_stop=None;self.observations=0;self.max_interval=0.;self.max_observation=0.
    def tick(self,**kwargs): self.observations+=1
    def close(self): pass


def h6_prepared(prefix):
    return SimpleNamespace(preparation_hash='2'*64,constant_coefficient=3.,extracted_identity_coefficient=0.,
        deterministic_fragment_indices=tuple(range(prefix)),randomized_block_indices=(),ld=prefix,
        deterministic_block_order=tuple(['one']+[f'df{i}' for i in range(prefix)]),
        deterministic_blocks=[{'id':i,'internal_phase':.125} for i in range(prefix+1)],num_system_qubits=2)


def h6_setup(root,output,control='symmetric_directional'):
    binding=dict(source='a'*40,hamiltonian='b'*64,df='c'*64,state='d'*64,input='e'*64,
        compiler='f'*64,fragment_order=[str(i) for i in range(19)])
    fingerprints={}
    for prefix,q in bridge.PILOT:
        prepared=h6_prepared(prefix)
        fingerprints[f'{prefix}/{q}']=d0.digest(bridge.blueprint(prepared,binding,prefix=prefix,q=q))
    binding['b0_fingerprints']=fingerprints
    contract=base_contract(bridge.SCOPE,root,output,binding);contract.update(control=control,max_instructions=100000)
    auth=authorization(contract,output)
    grant=g.authorization_gate(contract,auth,contract_sha=d0.digest(contract),authorization_sha=d0.digest(auth),
        output=output,scope=bridge.SCOPE,synthetic=True)
    record=dict(signals=dict(corrected=dict(real=.7,imag=.2),exact_truncated=dict(real=.6,imag=.1)),
        reference=dict(real=.5,imag=.3),allowance=dict(log_B_margin=1e-11),
        error_components=dict(kind='EMPIRICAL',certified=False,B0_signal=dict(real=.001,imag=.002),
            full_H_reference=dict(real=.003,imag=.004),signal_abs_upper=1.,
            evidence=dict(kind='ARTIFICIAL_SEPARATE_PATH_AND_REFERENCE_ERRORS')))
    fragments=[dict(id=str(i),G=((1,1j),(-1j,2)),**{'lambda':(-1. if i%2 else .5)}) for i in range(19)]
    return grant,record,fragments

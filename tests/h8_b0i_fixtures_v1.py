"""Tiny fabricated matrices/string circuits; no real scientific input."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


r=load('h8_fixture_runner',ROOT/'scripts/h8_b0i_family_runner_v1.py')
g=r.g;b=r.adapter_module.b
old=load('h8_previous_fixtures',ROOT/'tests/final_pre_review_fixtures_v1.py')
old_parallel=load('h8_previous_backend',ROOT/'tests/test_h6_b0i_parallel_compile_v1.py')
parallel=load('h8_fixture_parallel',ROOT/'scripts/h8_b0i_family_parallel_compile_v1.py')


class FakeMonitor(old.FakeMonitor):
    def __init__(self,grant):
        super().__init__(grant);self.child_cpu_seconds=0.


def authorization(c,out):
    value=old.authorization(c,out);value['canonical_contract_digest']=g.d0.digest(c);return value


def setup(root,out):
    binding=dict(source='b'*40,hamiltonian='1'*64,df='2'*64,state='3'*64,input='4'*64,
        compiler='5'*64,fragment_order=[str(i) for i in range(28)],b0_fingerprints={})
    c=old.base_contract(r.SCOPE,root,out,binding)
    c.update(operation_limits=deepcopy(r.LIMITS),execution_order=r.POLICY,workers=2,
        candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.family.REGISTRY],
        new_candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.family.REGISTRY],
        control='symmetric_directional',control_tolerance=1e-8,transpiled_instruction_cap=1000,max_instructions=1000,
        precision_domain=dict(epsilon=list(r.family.EPS),factors=list(r.family.FACTORS)),
        compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
        preparation_policy=b.RECIPE_POLICY,expected_profile_digest=g.d0.digest(c['profile']),
        native_provider=dict(SOURCE='b'*40,source_hashes={}),
        matched_B0=dict(approved=False,automatic_build=False),
        error_component_policy='SAVED_COMBINED_EMPIRICAL_U_UPPER_FOR_EACH_CONTRIBUTION',
        external_resource_monitor=dict(enabled=True,role='EXISTING_METADATA_LAUNCHER',interval_seconds=1.,maximum_staleness_seconds=5.),
        parallel_compile=dict(enabled=True,max_workers=2,internal_threads=1,automatic_retry=False,
            backend='OWNED_FORK_AFTER_CASE_CORRECTNESS_V1',parent_death_signal='SIGKILL',worker_cpus=[2,4],driver_cpu=16),
        fixture_skip=[])
    c['budget']['wall_seconds']=None;c['budget']['output_inode_cap']=2000
    control=root.parent/'control';control.mkdir();c['control_directory']=str(control)
    for p,q in r.family.REGISTRY:c['binding']['b0_fingerprints'][f'{p}/{q}']=g.d0.digest(b.recipe(c,p,q))
    signals=[]
    for p,q in r.family.REGISTRY:
        signals.append(dict(cell=dict(id=f'B0_p{p}_q{q}_2nd',method='B0',prefix=p,q=q,formula='2nd'),
            signals=dict(corrected=dict(real=.7,imag=.2),exact_truncated=dict(real=.7,imag=.2)),
            independent_forward_signals=dict(corrected=dict(real=.7,imag=.2)),reference=dict(real=.7,imag=.2),
            allowance=dict(real=1e-12,imag=1e-12,log_B_margin=0.,kind='EMPIRICAL',certified=False)))
    cells=[dict(id=f'B0_p{p}_q{q}_2nd',method='B0',prefix=p,q=q,formula='2nd') for p,q in r.family.REGISTRY]
    cells += [dict(id=f'B1_p28_q{q}_{form}',method='B1',prefix=28,q=q,formula=form) for q in (1,2,4,8) for form in ('2nd','4th')]
    cells += [dict(id=f'B2_p7_q{q}_2nd_r4_K{k}',method='B2',prefix=7,q=q,formula='2nd') for q in (2,4,8) for k in (2,4)]
    c['comparison_policy']=dict(candidate_ids=[x['id'] for x in cells])
    baseline=[]
    for cell in cells:
        for e in r.family.EPS:
            for factor in r.family.FACTORS:
                baseline.append(dict(cell_id=cell['id'],cell=cell,method=cell['method'],epsilon=e,factor=str(factor),
                    status='DERIVED_SAVED_SCALAR',N_total=10,
                    metrics={m:dict(G_point=80 if cell['method']=='B2' else 100) for m in r.family.METRICS},
                    sample_phase='confirmation' if cell['method']=='B2' else 'exploration',sample_count=32 if cell['method']=='B2' else 1))
    inputs=root.parent/'inputs';inputs.mkdir();mapping={};receipts=[]
    for name,obj in [('signal_summary.json',dict(signals=signals)),('d0_candidate_accounting.json',dict(H8_CONFIRMATION_UNION6=baseline))]:
        raw=json.dumps(obj).encode();path=inputs/name;path.write_bytes(raw);mapping[name]=str(path)
        receipts.append(dict(logical_name=name,bytes=len(raw),sha256=g.sha(raw)))
    c['input_receipts']=receipts
    return c,authorization(c,out),mapping


def prepared(prefix):
    order=tuple(f'df-fragment-{i}' for i in range(prefix))
    return SimpleNamespace(preparation_hash='6'*64,constant_coefficient=-.3,extracted_identity_coefficient=0.,
        deterministic_fragment_indices=tuple(range(prefix)),randomized_block_indices=(),ld=prefix,num_system_qubits=2,
        deterministic_block_order=order,deterministic_blocks=tuple(SimpleNamespace(block_id=x) for x in order))


class FakeAdapter:
    instances=[]
    def __init__(self,grant,verified,mapping,calls,guard):
        self.grant,self.verified,self.mapping,self.calls,self.guard=grant,verified,mapping,calls,guard
        self.builds=[];self.compiles=[];self.prepared_by_prefix={};self.closed=False;self.instances.append(self);calls.take('snapshot_load')
    def prepare(self,p,q):
        record=deepcopy(next(x for x in json.loads(self.verified['signal_summary.json'])['signals'] if x['cell']['id']==f'B0_p{p}_q{q}_2nd'))
        if [p,q] in self.grant.contract['fixture_skip']:record['signals']['corrected']['real']+=.5
        if p not in self.prepared_by_prefix:self.calls.take('prefix_preparation');self.prepared_by_prefix[p]=prepared(p)
        request=b.prepare(self.grant,prefix=p,q=q,binding=b.binding_for(self.grant,p,q),
            load_fragments=lambda _: [dict(id=str(i),G=((1.,0.),(0.,2.)),**{'lambda':0.}) for i in range(28)],
            load_prepared=lambda _:self.prepared_by_prefix[p],saved_record=record,
            scalar_input_error_upper=0.,rotation_roundoff_upper=1e-12,input_mapping_loader=lambda:self.mapping,guard=self.guard)
        return dict(request=request,independent=deepcopy(request['signal']['signals']['corrected']))
    def build(self,case):
        self.builds.append((case['prefix'],case['q']))
        return b.build(self.grant,case['request'],SimpleNamespace(
            build_deterministic_native=lambda *args,**kw:SimpleNamespace(circuit='ARTIFICIAL',scalar=kw['scalar']),
            build_native_hadamard_wrapper=lambda evolution,**kw:f"{case['prefix']}/{case['q']}/{kw['axis']}"),guard=self.guard)
    def validate(self,case,built):
        self.calls.take('probe_wrapper_copy',2);self.calls.take('control_probe',4)
        return dict(status='PASS',fingerprints={axis:self.fingerprint(w) for axis,w in built['wrappers'].items()})
    def fingerprint(self,w):return g.sha(w.encode())
    def compile(self,w,fp):
        self.compiles.append(w);self.compiler=None
        self.cost_api=SimpleNamespace(transpile_and_measure_cost=self.fabricated_cost)
        return r.adapter_module.NativeAdapter.compile(self,w,fp)
    def fabricated_cost(self,w,compiler,*,actual_circuit_fingerprint):
        return SimpleNamespace(**{x:10 for x in r.adapter_module.METRICS.values()},
            actual_circuit_fingerprint=actual_circuit_fingerprint,compiler_settings_hash=self.grant.contract['binding']['compiler'],
            transpiled_circuit=SimpleNamespace(data=[None]))
    def verify_bindings(self):pass
    def close(self):self.closed=True;self.prepared_by_prefix.clear()

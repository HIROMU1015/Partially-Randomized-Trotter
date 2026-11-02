"""Fabricated two-mode matrices, string circuits and mock process metadata."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

r = load('family_artificial_runner', ROOT/'scripts/h6_b0i_family_runner_v1.py')
old = load('family_original_fixtures', ROOT/'tests/final_pre_review_fixtures_v1.py')
old_parallel = load('family_scheduler_fixtures', ROOT/'tests/test_h6_b0i_parallel_compile_v1.py')
parallel = load('family_new_scheduler', ROOT/'scripts/h6_b0i_family_parallel_compile_v1.py')
g = r.g


def authorization(contract, output):
    value = old.authorization(contract, output)
    value['canonical_contract_digest'] = g.d0.digest(contract)
    return value


def setup(root, output):
    base, record, fragments = old.h6_setup(root, output)
    c = deepcopy(base.contract); c['scope'] = r.SCOPE
    science_binding={k:v for k,v in c['binding'].items() if k!='b0_fingerprints'}
    c['binding']['b0_fingerprints'] = {f'{p}/{q}':g.d0.digest(r.adapter_module.b.blueprint(
        old.h6_prepared(p), science_binding, prefix=p, q=q)) for p,q in r.family.REGISTRY}
    c.update(candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.family.REGISTRY],
        new_candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.family.NEW],
        operation_limits=deepcopy(r.LIMITS), control_tolerance=1e-8,
        compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
        precision_domain=dict(epsilon=list(r.family.EPS),factors=list(r.family.FACTORS)),
        matched_B0=dict(approved=False,automatic_build=False), workers=6,
        wall_policy='FUTURE_PHASE_TOTAL_WALL_NULL_CPU_SEPARATE',
        external_resource_monitor=dict(enabled=True,role='EXISTING_METADATA_LAUNCHER',
            interval_seconds=1.,maximum_staleness_seconds=5.),
        accepted_reuse_classifications=['REUSE_WITH_DECLARED_LIMITS'],
        reuse_binding=dict(contract_sha256='b'*64),expected_profile_digest=g.d0.digest(c['profile']),
        parallel_compile=dict(enabled=True,max_workers=6,internal_threads=1,
            backend='OWNED_FORK_AFTER_ALL_AB_V1',parent_death_signal='SIGKILL',automatic_retry=False,
            driver_cpu=16,worker_cpus=[2,4,5,6,8,9]),
        scalar_error_policy={f'{p}/{q}':dict(stored_model_coefficient_error_upper=0.,
            rotation_roundoff_upper=1e-12) for p,q in r.family.NEW})
    c['budget']['wall_seconds'] = None
    c['budget']['output_inode_cap'] = 1000
    control = root.parent/'control'; control.mkdir(); c['control_directory'] = str(control)
    record['signals']['corrected'] = dict(real=.5,imag=.3)
    record['error_components']['B0_signal'] = dict(real=1e-12,imag=1e-12)
    record['error_components']['full_H_reference'] = dict(real=1e-12,imag=1e-12)
    for fragment in fragments:
        fragment['lambda'] = 0.
        fragment['G'] = [[dict(real=complex(z).real,imag=complex(z).imag) for z in row] for row in fragment['G']]
    c['fixture_record'] = record; c['fixture_fragments'] = fragments; c['fixture_skip'] = []
    saved = dict(H6_CONFIRMATION_UNION8=[])
    cells = [dict(id=f'B0_p{p}_q{q}_2nd',method='B0',prefix=p,q=q,formula='2nd') for p,q in r.family.REGISTRY]
    cells += [dict(id=f'B1_p19_q{q}_{formula}',method='B1',prefix=19,q=q,formula=formula)
        for q in (1,2,4,8) for formula in ('2nd','4th')]
    cells += [dict(id=f'B2_p5_q{q}_2nd_r2_K{K}',method='B2',prefix=5,q=q,formula='2nd')
        for q in (1,2,4,8) for K in (2,4)]
    c['comparison_policy'] = dict(candidate_ids=[x['id'] for x in cells])
    for cell in cells:
        for eps in r.family.EPS:
            for factor in r.family.FACTORS:
                saved['H6_CONFIRMATION_UNION8'].append(dict(cell_id=cell['id'],method=cell['method'],
                    cell=cell,epsilon=eps,factor=str(factor),status='DERIVED_SAVED_SCALAR',
                    metrics={m:dict(G_point=100 if cell['method']!='B2' else 80) for m in r.family.METRICS},
                    N_total=2,sample_phase='confirmation' if cell['method']=='B2' else 'exploration',
                    sample_count=32 if cell['method']=='B2' else 1))
    payloads={'d0_candidate_accounting.json':saved}
    pilot=dict(SOURCE=r.family.OLD_SOURCE,compiler=c['binding']['compiler'],
        old_science_binding=deepcopy(c['binding']),cases=[])
    for p,q in r.family.REUSED:
        prior_signal=r.adapter_module.b.saved_signal_overlay(record,C=0.,C_error=0.,
            rotation_roundoff=1e-12,phase_roundoff=0.)
        prior_signal['cell']=dict(id=f'B0I_p{p}_q{q}_2nd')
        precision=[dict(epsilon=e,factor=f,accounting=g.d0.accounting(prior_signal,e,f))
            for e in r.family.EPS for f in r.family.FACTORS]
        fps={a:g.sha(f'{p}/{q}/{a}'.encode()) for a in ('cosine','sine')}
        pilot['cases'].append(dict(prefix=p,q=q,axes={a:dict(metrics={m:10 for m in r.family.METRICS},
            fingerprint=fps[a],compiler_hash=c['binding']['compiler'],worker=dict(native_exit_code=0)) for a in fps}))
        payloads[f'pilot_scalar_{p}_{q}.json']=dict(prefix=p,q=q,precision=precision,
            blueprint=r.adapter_module.b.blueprint(old.h6_prepared(p),science_binding,prefix=p,q=q),
            scalar=dict(full_Fock_identity=True,new_scientific_certificate=False),
            signal=dict(cell=dict(id=f'B0I_p{p}_q{q}_2nd',method='B0I',prefix=p,q=q,formula='2nd',T=.8),
                allowance=dict(kind='EMPIRICAL',certified=False)))
        payloads[f'pilot_correctness_{p}_{q}.json']=dict(status='PASS',fingerprints=fps,max_error=1e-12,
            reference_policy='SAVED_INDEPENDENT_FORWARD_OVERLAP_NOT_NEW_FULL_VECTOR_ORACLE')
    payloads.update({'pilot_results.json':pilot,
        'pilot_terminal.json':dict(status='COMPLETE_MANDATORY_STOP',first_stop=None),
        'pilot_native_exit.json':dict(shell_wait_status=0,same_parent_wait=True,original_identity_absent=True),
        'pilot_cleanup.json':dict(all_reaped=True,errors=[],workers=[dict(reaped=True,native_exit_code=0)]*6),
        'pilot_authorization.json':dict(SOURCE=r.family.OLD_SOURCE,approved_by_user=True,
            runtime_authorization=True,contract_sha256='b'*64,profile_digest=c['expected_profile_digest']),
        'pilot_launcher_wait.json':dict(worker_wait_status=0,recorder_status=0,first_launcher_stop=None)})
    mapping={}; receipts=[]
    inputs=root.parent/'input';inputs.mkdir()
    for name,value in payloads.items():
        raw=json.dumps(value).encode();path=inputs/name;path.write_bytes(raw);mapping[name]=str(path)
        receipts.append(dict(logical_name=name,bytes=len(raw),sha256=g.sha(raw)))
    c['input_receipts']=receipts
    return c,authorization(c,output),mapping,payloads


class FakeAdapter:
    instances=[]
    def __init__(self,grant,verified,mapping,calls,guard):
        self.grant,self.mapping,self.calls,self.guard=grant,mapping,calls,guard
        self.builds=[];self.compiles=[];self.closed=False;self.instances.append(self);calls.take('snapshot_load')
    def prepare(self,p,q):
        b=r.adapter_module.b;record=deepcopy(self.grant.contract['fixture_record'])
        if [p,q] in self.grant.contract['fixture_skip']:
            record['signals']['corrected']['real']+=.4
        fragments=deepcopy(self.grant.contract['fixture_fragments'])
        for fragment in fragments:
            fragment['G']=tuple(tuple(complex(z['real'],z['imag']) for z in row) for row in fragment['G'])
        req=b.prepare(self.grant,prefix=p,q=q,binding=b.binding_for(self.grant,p,q),
            load_fragments=lambda _:fragments,load_prepared=lambda _:old.h6_prepared(p),
            saved_record=record,scalar_input_error_upper=0.,rotation_roundoff_upper=1e-12,
            input_mapping_loader=lambda:self.mapping)
        return dict(request=req,independent=deepcopy(req['signal']['signals']['corrected']))
    def build(self,case):
        self.builds.append((case['prefix'],case['q']))
        return dict(wrappers={a:f"{case['prefix']}/{case['q']}/{a}" for a in ('cosine','sine')})
    def validate(self,case,built):
        self.calls.take('probe_wrapper_copy',2);self.calls.take('control_probe',4)
        return dict(status='PASS',fingerprints={a:self.fingerprint(w) for a,w in built['wrappers'].items()})
    def fingerprint(self,w):return g.sha(w.encode())
    def compile(self,w,fp):
        self.compiles.append(w)
        return dict(metrics={m:10 for m in r.family.METRICS},fingerprint=fp,
            compiler_hash=self.grant.contract['binding']['compiler'])
    def verify_bindings(self):pass
    def close(self):self.closed=True

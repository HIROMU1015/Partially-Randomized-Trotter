"""Entirely fabricated two-mode integration data; no saved scientific inputs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value

ROOT=Path(__file__).resolve().parents[1]
r=load('artificial_h6_pilot_runner',ROOT/'scripts/h6_b0i_pilot_runner_v1.py')
f=load('artificial_h6_pilot_old_fixtures',ROOT/'tests/final_pre_review_fixtures_v1.py')
b=r.adapter_module.b


def setup(root,output):
    grant,record,fragments=f.h6_setup(root,output)
    contract=grant.contract
    contract.update(candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.PILOT],
        operation_limits=deepcopy(r.LIMITS),workers=1,
        compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
        control_tolerance=1e-8,precision_domain=dict(epsilon=[.05,.01,.005,.001,.0005],factors=[1.,10.]),
        matched_B0=dict(automatic_build=False,approved=False),
        scalar_error_policy={f'{p}/{q}':dict(stored_model_coefficient_error_upper=0.,rotation_roundoff_upper=1e-12) for p,q in r.PILOT})
    # Exact zero bias permits every artificial precision without changing rules.
    record['signals']['corrected']=dict(real=.5,imag=.3)
    record['error_components']['B0_signal']=dict(real=1e-12,imag=1e-12)
    record['error_components']['full_H_reference']=dict(real=1e-12,imag=1e-12)
    for fragment in fragments:
        fragment['G']=[[dict(real=complex(z).real,imag=complex(z).imag) for z in row] for row in fragment['G']]
    contract['fixture_record']=record;contract['fixture_fragments']=fragments
    return contract,f.authorization(contract,output)


class FakeAdapter:
    instances=[]
    def __init__(self,grant,verified,mapping,calls,guard):
        self.grant=grant;self.calls=calls;self.guard=guard;self.closed=False;self.compiles=[];self.builds=[]
        self.native_calls=[];self.instances.append(self);calls.take('snapshot_load')
        self.native=SimpleNamespace(build_deterministic_native=self.evolution,
            build_native_hadamard_wrapper=self.wrapper)
    def evolution(self,blocks,**kw):
        self.native_calls.append(('evolution',kw['scalar']));return 'ARTIFICIAL_EVOLUTION'
    def wrapper(self,evolution,**kw):return 'ARTIFICIAL_'+kw['axis']
    def prepare(self,p,q):
        record=deepcopy(self.grant.contract['fixture_record']);fragments=deepcopy(self.grant.contract['fixture_fragments'])
        for fragment in fragments:
            fragment['G']=tuple(tuple(complex(z['real'],z['imag']) for z in row) for row in fragment['G'])
        request=b.prepare(self.grant,prefix=p,q=q,binding=b.binding_for(self.grant,p,q),
            load_fragments=lambda _:fragments,load_prepared=lambda _:f.h6_prepared(p),saved_record=record,
            scalar_input_error_upper=0.,rotation_roundoff_upper=1e-12)
        return dict(request=request,independent=deepcopy(request['signal']['signals']['corrected']))
    def build(self,case):
        self.builds.append(case['prefix']);built=b.build(self.grant,case['request'],self.native)
        built['wrappers']={a:f'{case["prefix"]}/{case["q"]}/{a}' for a in ('cosine','sine')};return built
    def validate(self,case,built):
        self.calls.take('control_probe',4);self.calls.take('probe_wrapper_copy',2)
        return dict(status='PASS',max_error=0.,certified=False,
            fingerprints={a:self.fingerprint(w) for a,w in built['wrappers'].items()})
    def fingerprint(self,wrapper):return r.g.sha(wrapper.encode())
    def compile(self,wrapper,expected_fingerprint):
        self.compiles.append(wrapper)
        return dict(metrics={m:10 for m in r.adapter_module.METRICS},fingerprint=expected_fingerprint,
            compiler_hash=self.grant.contract['binding']['compiler'],wall_seconds=0.)
    def legacy_b0(self,p,q):return dict(cosine=dict(status='MISSING',metrics=None),sine=dict(status='MISSING',metrics=None))
    def verify_bindings(self):pass
    def close(self):self.closed=True

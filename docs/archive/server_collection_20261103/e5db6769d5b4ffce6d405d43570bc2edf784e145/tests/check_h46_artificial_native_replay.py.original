"""One tiny artificial native replay. No real input, RNG draws or transpile."""
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test_execution',ROOT/'scripts/h46_cost_execution.py')
x=importlib.util.module_from_spec(s);sys.modules[s.name]=x;s.loader.exec_module(x)
same=x.load('_test_same',ROOT/'scripts/h46_same_event.py')
provider=sys.argv[1]
rte,ops=x.provider_modules(dict(provider=provider,versions={'qiskit':'1.3.0','numpy':'1.26.4','scipy':'1.14.1'}))
import numpy as np
ham_api=importlib.import_module('trotterlib.df_hamiltonian')
prep_api=importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')
native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
ham=ham_api.DFHamiltonian(.1,np.diag([.2,.1]).astype(complex),np.array([.3]),(np.diag([.7,.4]).astype(complex),),{})
p=prep_api._prepare(ham,0);meta=x.from_live(p)
cell=dict(id='ARTIFICIAL',method='B2',prefix=0,q=1,r=1,K=2,formula='2nd');task=dict(seed=0)
tolerance=importlib.import_module('trottertracks.resource_applicability.ax2b_molecular_ports_v3')._explicit_cutoff_tolerance
config,dist=rte.make_rte_config(p.rte_preparation.symbolic_tail,evolution_time=.8,rte_steps=1,finite_taylor_order=2,truncation_tolerance=tolerance(p.exact_rte_lambda_r*.8,2),seed=0)
c=next(c for c in p.rte_preparation.symbolic_tail.components if not c.is_identity)
app=rte.RTEEventApplication(application_index=0,role='rotation',component_id=c.component_id,coefficient_abs=c.coefficient_abs,
    coefficient_sign=c.coefficient_sign,is_identity=c.is_identity,df_fragment_id=c.df_fragment_id,basis_id=c.basis_id,
    diagonal_pauli_support=c.diagonal_pauli_support,basis_change_operations=c.basis_change_operations,basis_hash=c.basis_hash)
coefficient=dist.unnormalized_order_weights[0]*c.probability
e=rte.RTEEvent(taylor_order=0,rotation_component_id=c.component_id,product_component_ids=(),selected_component_ids=(c.component_id,),
    df_fragment_ids=(c.df_fragment_id,),event_probability=coefficient/dist.exact_finite_distribution,event_coefficient=coefficient,
    phase=1+0j,rotation_angle=math.atan(config.dimensionless_step_time),event_normalization=dist.exact_finite_distribution,
    basis_id=c.basis_id,basis_reuse_intervals=(),application_sequence=(app,),basis_hash=c.basis_hash)
raw=json.loads(json.dumps([[e.to_dict()]]));trajectory=dict(events=raw,event_digest=x.lib.digest(raw),finite_distribution=dist.to_dict())
events,steps=same.replay_requests(x,p,cell,task,trajectory,rte)
theory=same.paired_theory(x,meta,cell,events,ops)
evolution=native.partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=10000)
for axis in ('cosine','sine'):
    wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,max_instructions=10000)
    measured=same.raw_lowered(x,wrapper)
    wanted=next(r['cost'] for r in theory if r['model']=='NATIVE_MIXED' and r['axis']==axis)
    assert measured['cost']==wanted,(measured,wanted)
old=dict(cell_id='fake',axis='cosine',control='symmetric_directional',fingerprint='f',event_digest='e',compiler_hash='c')
assert same.reusable(old,candidate='fake',axis='cosine',fingerprint='f',event_digest='e',compiler='c')
assert not same.reusable(old,candidate='fake',axis='sine',fingerprint='f',event_digest='e',compiler='c')
bad=dict(trajectory,event_digest='bad')
try:same.replay_requests(x,p,cell,task,bad,rte)
except ValueError:pass
else:raise AssertionError('changed event digest accepted')
print(json.dumps(dict(status='PASS_ARTIFICIAL_NATIVE_REPLAY',artificial_qubits=3,axes=2,new_real_trajectories=0,random_draws=0,compile=0,
                     checks=['event roundtrip','finite law','request validation','native mixed exact lowered counts both axes','saved digest reuse','axis mismatch rejection','event mismatch rejection'])))

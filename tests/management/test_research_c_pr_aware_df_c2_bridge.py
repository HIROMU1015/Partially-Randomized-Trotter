"""Invented molecular arrays/CAR, exact proof inputs and mock provenance only."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_c2_bridge as m
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks import c_pr_aware_df_stage_c2 as t


def artificial(role='baseline'):
    # Same fixture helper already exercises general-K/CAR/native in B1 tests.
    from test_research_c_pr_aware_df_stage_b import artificial as make,sector_generators
    old,h,g,J,U=make(n=4,rank=8)
    arm=b.prepare_arm(h,g,old.model.constant,J,U,role=role,
        source_identity='synthetic:'+role,input_identity='mock',ld=2)
    ctx,G=sector_generators(arm,h,g)
    tail,config,dist=b.configure(arm,Fraction(1,40),16,0)
    from trottertracks.pr_dictionary_validation.prototype import atom_record
    saved=dict(atoms_catalog=[atom_record(a) for a in arm.plan.atoms],
        components=[{k:getattr(c,k) for k in ('component_id','probability','coefficient_abs','coefficient_sign','basis_id','df_fragment_id')} for c in tail.components],
        native_policy=b.NATIVE_POLICY,lambda_R=tail.lambda_r,identity=arm.plan.extracted_identity,
        configuration=dict(tau=config.dimensionless_step_time,r=16,K=0),
        finite_distribution=dict(orders=[0],order_probabilities=[1.],normalization=dist.exact_finite_distribution))
    saved=json.loads(json.dumps(saved))
    table=t.analyze_table([str(abs(a.coefficient)) for a in arm.plan.atoms],[c.probability for c in tail.components])
    kernels=b.kernels(arm,G,Fraction(1,40),config)
    return dict(arm=arm,ctx=ctx,G=np.array(G),tail=tail,config=config,dist=dist,saved=saved,table=table,
                packet=dict(h=h,g=g,constant=np.asarray(old.model.constant)),data=dict(J=J,U=U),kernels=kernels)


@pytest.fixture(scope='module')
def invented():return artificial()


def test_import_passive():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_c2_bridge as m;assert m.MAX_CALLS==47;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"],cwd=ROOT,check=True)


def test_replay_general_K_sign_spin_frame_identity_and_K0(invented):
    x=invented;a,tail,config,dist=m.replay(x['packet'],x['data'],x['saved'],dict(role='baseline',export_sha256='synthetic:baseline'),'mock')
    assert len(a.plan.atoms)==216 and a.plan.ld==2 and config.rte_steps==16
    assert any(v.coefficient<0 for v in a.plan.atoms)
    assert all(v.kind=='FRAME_Z' and len(v.support) in (1,2) for v in a.plan.atoms)
    assert dist.orders==(0,) and a.plan.extracted_identity!=0
    ops=b.atom_operators(x['ctx'],a)
    report=m.atom_audit(a,tail,ops,x['G'],x['table'])
    assert report['status']=='FINITE_SECTOR_ATOM_NUMERICAL_PASS'
    assert report['raw_tail']['max_entry']<1e-12 and report['reference_uncertainty'] is None
    assert any(i%2 for atom in a.plan.atoms for i in atom.support)
    assert any(not i%2 for atom in a.plan.atoms for i in atom.support)


@pytest.mark.parametrize('bad',['sign','frame','normalization','order'])
def test_replay_rejects_correspondence_faults(invented,bad):
    x=invented;s=deepcopy(x['saved'])
    if bad=='sign':s['components'][0]['coefficient_sign']*=-1
    if bad=='frame':s['atoms_catalog'][0]['frame_id']='WRONG'
    if bad=='normalization':s['finite_distribution']['normalization']*=1.001
    if bad=='order':s['finite_distribution']['orders']=[0,2]
    with pytest.raises(ValueError):m.replay(x['packet'],x['data'],s,dict(role='baseline',export_sha256='synthetic:baseline'),'mock')


def test_K0_signed_event_mean_car_raw_corrected_identity_phase(invented):
    x=invented;ops=b.atom_operators(x['ctx'],x['arm'])
    mean=m.event_mean(x['arm'],x['tail'],x['dist'],ops,x['table'])
    head=m.heads(x['arm'],x['G'])
    context=m.kernel_bridge(x['arm'],x['config'],mean,head,dict(generators=x['G'],corrected=x['kernels']['corrected']))
    assert mean['report']['mean_identity']['max_entry']<1e-12
    assert context['report']['atom_vs_raw']['max_entry']<1e-12
    assert context['report']['rigorous_bridge_bound'] is None
    assert context['report']['tail_identity_phase_parts']!=[1.,0.]
    # Removing extracted identity phase or reversing Gaussian direction is a correctness error.
    bad=deepcopy(x['table']);bad['ideal_probabilities']=list(reversed(bad['ideal_probabilities']))
    with pytest.raises(ValueError):m.atom_audit(x['arm'],x['tail'],ops,x['G'],bad)


def raw_row(context,psi,m):
    from trotterlib.rte import finite_rte_attenuation
    A=finite_rte_attenuation(context['config'],tail_evolutions=4*(1<<m))
    z=complex(np.vdot(psi,A*np.linalg.matrix_power(context['saved_corrected'],4*(1<<m))@psi))
    return dict(m=m,reference=dict(attenuation=A,raw_signal_parts=[z.real,z.imag],pf_operator_error=.002,rte_corrected_operator_error=.003))


def test_round_bridge_preserves_observed_vs_bound_and_saved_signal(invented):
    x=invented;ops=b.atom_operators(x['ctx'],x['arm']);mean=m.event_mean(x['arm'],x['tail'],x['dist'],ops,x['table'])
    context=m.kernel_bridge(x['arm'],x['config'],mean,m.heads(x['arm'],x['G']),dict(generators=x['G'],corrected=x['kernels']['corrected']))
    psi=np.linalg.eigh(sum(x['G']))[1][:,0];saved=raw_row(context,psi,12)
    law=t.law_round(x['table'],12);r=m.round_bridge(context,12,saved,psi,law)
    assert r['operator_distance_observed']>=0 and r['bridge_raw_signal_bound'] is None
    assert r['guaranteed_shots'] is r['numerical_reference_uncertainty'] is None
    saved['reference']['raw_signal_parts'][0]+=.01
    with pytest.raises(ValueError,match='SAVED_RAW_SIGNAL'):m.round_bridge(context,12,saved,psi,law)


def proof(m=0,bound=1e-7,**kw):return dict(m=m,bound=bound,kind='analytic_bound',norm='raw_complex_signal',provenance='artificial exact argument',**kw)


def schema(**kw):
    vals=dict(m=0,attenuation=.5,pf=.01,rte=.02,
        categorical=proof(),bridge=proof(reference_uncertainty_included=False),
        synthesis=proof(compiler_coverage='all_final_whole_kernel_P0_outputs'),
        reference=proof(),state_alias=True,random_model=t.IID)
    vals.update(kw);return m.signal_schema(**vals)


def test_raw_units_reference_no_double_count_and_complete_conditional_model():
    r=schema();assert r['status']=='CONDITIONAL_MODEL_INPUTS_COMPLETE'
    assert r['conditional_total_raw']==pytest.approx(.5*(.01+.02)+4e-7)
    assert r['error_evidence'] is r['actual_implemented_law_error'] is r['guaranteed_total'] is None
    law=t.law_round(dict(status='EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL',eta_exact='1/1000000000000'),0)
    cat=m.categorical_input(law,'mock fixed C2B')
    assert cat['actual_implemented_law_error'] is None and not cat['reference_bridge_included']
    assert schema(categorical=cat)['status']=='CONDITIONAL_MODEL_INPUTS_COMPLETE'


@pytest.mark.parametrize('name',['bridge','synthesis','reference'])
def test_missing_bound_never_zero_filled(name):
    r=schema(**{name:None});assert name in r['missing'] and r['conditional_total_raw'] is None
    observed=proof();observed['kind']='binary64_observed_residual'
    r=schema(**{name:observed});assert r['conditional_total_raw'] is None


def test_whole_P0_phase_reference_and_IID_scope():
    with pytest.raises(ValueError,match='COMPILER'):schema(synthesis=proof(compiler_coverage='event_P0'))
    with pytest.raises(ValueError,match='DOUBLE_COUNT'):schema(bridge=proof(reference_uncertainty_included=True))
    assert schema(random_model='PCG64_seed_separated')['conditional_total_raw'] is None
    req=m.primitive_requirements();assert req['per_arm']['signed_atoms']==216
    assert req['all_law_synthesis_bound'] is None and not req['event_P0_sum_is_whole_P0_bound']
    assert m.uniform_synthesis_preview([None],{k:None for k in ('head','X','Y')},12)['whole_law_synthesis'] is None
    fake=dict(kind='sample_max_residual')
    with pytest.raises(ValueError):m.uniform_synthesis_preview([fake],{k:fake for k in ('head','X','Y')},12)


def test_actual_raw_compile_relative_phase_on_artificial_kernel():
    # Reuse the established artificial full controlled/P0/adjoint coverage.
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    lock=SharedQiskitCalls(lambda k,l,fn:fn())
    qc=QuantumCircuit(2);qc.ry(.13,0);qc.cx(1,0);qc.rz(-.02,0);qc.cx(1,0);qc.ry(-.13,0);qc.p(.17,1)
    compiled=lock('compile','invented',lambda:compiled_wrapper(qc,CompilerProfile()))
    R=Operator(qc).data;C=Operator(compiled).data;assert np.max(abs(R-C))<1e-10
    broken=qc.copy();broken.data.pop();assert np.max(abs(Operator(broken).data-R))>.1
    with lock.adjoint_builds():adj=b.native.actual_adjoint(compiled)
    assert np.max(abs(Operator(adj).data@C-np.eye(4)))<1e-10


def synthetic_bundle(root):
    from trottertracks import c_pr_aware_df_stage_b_runner as old
    records={};exports={};references={};grounds={}
    records['B1_result']=dict(status='B1_MECHANISM_EVALUATION_COMPLETE',arms={})
    records['C2B_result']=dict(status='C2B_CONDITIONAL_IID_TABLE_ANALYSIS_COMPLETE',arms={},actual_implemented_law_error=None,whole_law_synthesis=None,guaranteed_shots=None,guaranteed_total=None)
    for name,count,commit in [('B1',141,'7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'),('C2B',35,'3c1bb2d9ce46c29205594a42b090c0811fbafd27')]:
        records[name+'_STOP']=dict(failed=False,scientific_units=count,retry=0,resume=0,mandatory_stop=True)
        records[name+'_SOURCE']=dict(execution_commit=commit)
    for arm in m.ARMS:
        x=artificial(arm);exports[arm]=arm+'.npz';old.save_arrays(root/exports[arm],**x['data'])
        references[arm]=arm+'_ref.npz';old.save_arrays(root/references[arm],generators=x['G'],corrected=x['kernels']['corrected'],full_input_indices=b.stage1.bit_reverse_indices(8)[x['ctx'].sector.basis_indices])
        psi=np.linalg.eigh(sum(x['G']))[1][:,0];grounds[arm]=arm+'_psi.npz';old.save_arrays(root/grounds[arm],sector_state=psi)
        context=dict(config=x['config'],saved_corrected=x['kernels']['corrected'])
        records['B1_result']['arms'][arm]=dict(rounds=[raw_row(context,psi,i) for i in range(13)],classification=dict(status='POTENTIALLY_FEASIBLE'))
        records[arm+'_ground']=dict(status='PASS');records[arm+'_tail']=x['saved']
        records['C2B_result']['arms'][arm]=dict(table=x['table'],preview=dict(conditional_law_rows=[t.law_round(x['table'],i) for i in range(13)]))
    old.save_arrays(root/'input.npz',**x['packet']);paths={}
    for k,v in records.items():paths[k]=k+'.json';m.io.save(root/paths[k],v)
    return dict(records=paths,exports=exports,references=references,grounds=grounds,stage_A_input='input.npz',
                saved_inputs={p:m.io.sha(root/p) for p in [*paths.values(),*exports.values(),*references.values(),*grounds.values(),'input.npz']},original_input_sha256='mock')


def test_artificial_47call_E2E_mock_and_correctness_STOP(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    c=synthetic_bundle(tmp_path);out=tmp_path/'out';out.mkdir();call=Calls(out,m.LIMITS,m.MAX_CALLS)
    r=m.execute(tmp_path,c,out,call)
    assert m.MAX_CALLS==47 and dict(call.counts)==m.LIMITS
    assert r['scientific_units']==47 and r['guaranteed_total'] is r['whole_law_synthesis'] is None
    assert len(list(out.glob('*_m??.json')))==26
    # B1 source mismatch is detected before any molecular array call.
    p=tmp_path/c['records']['B1_SOURCE'];p.write_text('{}');out2=tmp_path/'fail';out2.mkdir();bad=Calls(out2,m.LIMITS,m.MAX_CALLS)
    with pytest.raises((ValueError,KeyError)):m.execute(tmp_path,c,out2,bad)
    assert bad.stop.is_set() and bad.counts['input_load']==0


def test_preflight_source_and_input_guard_fresh_STOP(tmp_path,monkeypatch):
    c=synthetic_bundle(tmp_path);m.io.save(tmp_path/'source.json',dict(files={}))
    c.update(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT_DRAFT',implementation=m.VERSION,
        policy=m.POLICY,allowances=m.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,retry=0,resume=0,source_manifest='source.json',source_manifest_sha256=m.io.sha(tmp_path/'source.json'),packages={})
    m.io.save(tmp_path/'contract.json',c)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'mock_source\n')
    # Scope/STOP wrapper with numerical execution mocked; numerical endpoint has its own E2E above.
    monkeypatch.setattr(m,'execute',lambda *a:None)
    m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json');assert stop['mandatory_stop'] and not stop['failed']
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
    (tmp_path/c['exports']['baseline']).write_bytes(b'broken synthetic')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'source.json').write_text('{}')
    with pytest.raises(ValueError,match='SOURCE_MANIFEST'):m.preflight(tmp_path,tmp_path/'contract.json')


def test_sector_bit_mapping_mismatch_stops_before_atom_build(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    c=synthetic_bundle(tmp_path);p=tmp_path/c['references']['baseline']
    with np.load(p,allow_pickle=False) as a:values={k:a[k] for k in a.files}
    values['full_input_indices']=values['full_input_indices'][::-1]
    p.unlink() # This invented fixture only.
    from trottertracks.c_pr_aware_df_stage_b_runner import save_arrays
    save_arrays(p,**values)
    out=tmp_path/'out';out.mkdir();call=Calls(out,m.LIMITS,m.MAX_CALLS)
    with pytest.raises(ValueError,match='BIT_ORDER'):m.execute(tmp_path,c,out,call)
    assert call.stop.is_set() and call.counts['atom_context']==0

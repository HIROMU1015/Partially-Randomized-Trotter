"""Artificial molecules/circuits only; no real H4 decode/build/compile."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h4_native_fair as m
from trottertracks import c1_h6_native_precision as precision
from trottertracks import c1_stage3_accuracy as accuracy


def contract_fixture(tmp_path,monkeypatch):
    root=tmp_path/'ARTIFICIAL';root.mkdir();inp=root/'packet.npz';inp.write_bytes(b'ARTIFICIAL_NOT_NUMERIC')
    monkeypatch.setattr(m.io,'P_H4_SHA256',m.io.sha(inp))
    records={k:k+'.json' for k in ['CDF_PF_s1','CDF_PF_s4','CDF_prepared','CDF_STOP','CDF_SOURCE',
        'CDF_stage1','CDF_provenance','XDF_result','XDF_STOP','XDF_SOURCE']}
    fields=dict(export='export.npz',fit_SOURCE='fit_SOURCE.json',fit_metadata='fit_metadata.json',CDF_sector='CDF_sector.npz',
        XDF_generators='XDF_generators.npz',CDF_operators=[f'C{i}.npz' for i in range(9)],records=records,
        ground_states={a:a+'_ground.npz' for a in m.ARMS},XDF_kernels={str(s):f'XDF_s{s}.npz' for s in (1,4)})
    paths=[fields[k] for k in ['export','fit_SOURCE','fit_metadata','CDF_sector','XDF_generators']]
    paths += fields['CDF_operators']+list(records.values())+list(fields['ground_states'].values())+list(fields['XDF_kernels'].values())
    for p in paths:(root/p).write_bytes(b'ARTIFICIAL_ONLY')
    (root/'code.txt').write_text('ARTIFICIAL CODE\n');m.io.save(root/'SOURCE.json',dict(files={'code.txt':m.io.sha(root/'code.txt')}))
    c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=copy.deepcopy(m.CONDITIONS),
        allowances=copy.deepcopy(m.ALLOWANCES),compiler=m.profile_dict(),gaussian=m.GAUSSIAN,
        compiler_phase=m.PHASE,synthesis=m.synthesis.VERSION,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        original_input=dict(sha256=m.io.sha(inp)),saved_inputs={p:m.io.sha(root/p) for p in paths},
        packages={},fresh_output=m.FAMILY+'/<execution-date>/run01',lineage=dict(XDF_PF_SOURCE='ARTIFICIAL'),
        **fields,**{k:0 for k in m.FORBIDDEN})
    m.io.save(root/'contract.json',c);return root,c,inp


def artificial_data():
    # 4 system modes / 32 controlled components,8 artificial square frames.
    n=2;modes=4;car=m.stage.car_bilinears(modes);idx=m.stage.sector_indices(2,1,1)
    h=np.array([[.2,1e-9],[1e-9,-.1]]);lam=np.array([(-1.)**i*.01/(i+1) for i in range(10)])
    G=[];J=[];U=[]
    for i in range(10):
        theta=[1e-9,np.pi-1e-12,.17,.31][i%4]
        v=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        v[:,0]*=(-1)**i;eta=np.array([.3+.01*i,-.2]);g=v@np.diag(eta)@v.T
        G.append(np.kron(g,np.eye(2)));J.append(2*lam[i]*np.outer(eta,eta));U.append(v)
    p=dict(df_one_body=np.kron(h,np.eye(2)).astype(complex),df_lambdas=lam,
        df_g_matrices=np.asarray(G,dtype=complex),nuclear_repulsion=np.array(.37))
    gens=[.37*np.eye(16)+m.native.xdf_car_generator(p['df_one_body'],None,car)]+[
        m.native.xdf_car_generator(G[i],lam[i],car) for i in range(8)]
    sector=[g[np.ix_(idx,idx)] for g in gens];psi=np.linalg.eigh(sum(sector))[1][:,0]
    kernels={s:m.reference._ordered_exponentials(gens,m.native.ordered_schedule(9,m.Fraction(1,10*s)))[np.ix_(idx,idx)] for s in (1,4)}
    data=dict(packet=p,exported=dict(kappa=h,diag_coulomb_mats=np.asarray(J[:8]),orbital_rotations=np.asarray(U[:8]),constant=np.array(.37)),
        metadata=dict(actual_rank=8,frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)]),
        CDF=gens,XDF_sector=np.asarray(sector),CDF_sector=sum(sector),grounds={a:psi.copy() for a in m.ARMS},
        XDF_kernels=kernels,modes=modes,indices=idx)
    verified=dict(status='PASS',DF_bounds_Ha={a:1e-5 for a in m.ARMS},
        rows={(a,s):[dict(m=i,pf_operator_error=.001,attenuation=1.,rte_corrected_operator_error=0.) for i in range(13)] for a,s in m.JOBS})
    return data,verified


def prepare_fake_run(tmp_path,monkeypatch):
    root,c,inp=contract_fixture(tmp_path,monkeypatch);data,verified=artificial_data()
    kinds=['input_load','export_load','evidence_load','cdf_operator_load','sector_generator_load','cdf_sector_load','ground_load','saved_kernel_load']
    def load(root,c,input_path,call):
        for kind in kinds:
            for i in range(m.LIMITS[kind]):call(kind,str(i),lambda:None)
        return data
    monkeypatch.setattr(m,'load_data',load);monkeypatch.setattr(m,'identity',lambda *args:verified)
    original=subprocess.check_output
    def git(command,*args,**kwargs):
        return 'ARTIFICIAL_SOURCE\n' if command[:2]==['git','rev-parse'] else original(command,*args,**kwargs)
    monkeypatch.setattr(m.subprocess,'check_output',git)
    output=root/m.FAMILY/'2026-10-26/run01'
    return root,c,inp,data,verified,output


def test_passive_import_and_preflight_never_decode_or_build(tmp_path,monkeypatch):
    root,c,inp=contract_fixture(tmp_path,monkeypatch)
    with patch.object(np,'load',side_effect=AssertionError('NO_SCIENCE')):
        assert m.preflight(root,root/'contract.json',inp)==c
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'src');from trottertracks import c1_h4_native_fair;assert not any(k in sys.modules for k in ['numpy','qiskit','scipy','ffsim'])"],cwd=ROOT,check=True)
    assert not (root/m.FAMILY).exists() and m.MAX_CALLS==305


def test_artificial_end_to_end_all_calls_QPY_full_coverage_and_rounds(tmp_path,monkeypatch):
    root,c,inp,data,verified,out=prepare_fake_run(tmp_path,monkeypatch)
    with patch('trottertracks.pr_dictionary_validation.circuits.native_gaussian',side_effect=AssertionError('OLD_GAUSSIAN_FORBIDDEN')), \
         patch.object(m.stage,'fit_once',side_effect=AssertionError('NO_FIT')), \
         patch.object(m.stage,'physical_sector',side_effect=AssertionError('NO_NEW_PHYSICAL')), \
         patch.object(accuracy,'assess_round',side_effect=AssertionError('NO_SHOTS_OR_CA')), \
         patch.object(accuracy,'assess_energy',side_effect=AssertionError('NO_ENERGY_ASSESSMENT')):
        result=m.run(root,root/'contract.json',inp,out)
    assert result['status']=='H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE'
    stop=m.io.read(out/'STOP.json');assert stop['scientific_units']==m.MAX_CALLS==305
    assert stop['call_counts']==m.LIMITS and not stop['failed'] and stop['retry']==stop['resume']==0
    assert len(list((out/'circuits').glob('*.qpy')))==40
    assert len(list((out/'operators').glob('*.npz')))==18
    assert len(result['local_reports'])==18 and len(result['shells'])==3
    assert len(m.io.read(out/'gaussian_precision.json'))==18
    for key,value in result['candidates'].items():
        assert value['segment']['input_dimension']==value['segment']['output_dimension']==32
        assert value['segment']['compiled_reference_frobenius']<1e-10
        assert len(value['rounds'])==13 and value['rounds'][-1]['occurrences']==value['s']*4096
        assert all(x['status']=='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL' and x['CA_achieved'] is None for x in value['rounds'])
        assert value['XY']['compiled_reference']['error']<2e-10
    assert result['CA_achieved'] is None and len(result['comparisons'])==2
    with pytest.raises(ValueError,match='FRESH'):m.run(root,root/'contract.json',inp,out)
    with pytest.raises(ValueError):m.fresh_output(root,out.parent/'run02')


@pytest.mark.parametrize('defect',['Gaussian','phase','order','compile'])
def test_artificial_correctness_FAIL_stops_and_preserves_partial(tmp_path,monkeypatch,defect):
    root,c,inp,data,verified,out=prepare_fake_run(tmp_path,monkeypatch)
    if defect=='Gaussian':
        original=precision.gaussian
        monkeypatch.setattr(precision,'gaussian',lambda u:original(np.asarray(u).conj().T))
    if defect=='phase':
        original=m.native.NativeBuilder.local
        def phase(self,*a,**k):
            qc=original(self,*a,**k);qc.p(.03,qc.num_qubits-1);return qc
        monkeypatch.setattr(m.native.NativeBuilder,'local',phase)
    if defect=='order':
        original=m.prepare_models
        def reversed_models(*a):
            models=original(*a);x=models['CDF'];models['CDF']=replace(x,blocks=(x.blocks[0],*x.blocks[:0:-1]));return models
        monkeypatch.setattr(m,'prepare_models',reversed_models)
    if defect=='compile':
        original=m.compiled_wrapper
        def wrong(*a):
            qc=original(*a);qc.global_phase+=.03;return qc
        monkeypatch.setattr(m,'compiled_wrapper',wrong)
    with pytest.raises(ValueError):m.run(root,root/'contract.json',inp,out)
    assert m.io.read(out/'STOP.json')['failed']
    assert m.io.read(out/'STOP.json')['scientific_units']<305
    assert (out/'failure.json').exists() and (out/'SOURCE.json').exists()
    assert not (out/'result.json').exists()


@pytest.mark.parametrize('status',['INFEASIBLE_REGISTERED_BOUND','UNRESOLVED_SYNTHESIS_OR_PF'])
def test_numerical_not_met_unknown_are_not_correctness_fail(tmp_path,monkeypatch,status):
    root,c,inp,data,verified,out=prepare_fake_run(tmp_path,monkeypatch)
    original=m.synthesis.propagate
    def row(*a,**k):
        if k['s']==1:k['saved_pf_error']=.9 if status.startswith('INFEASIBLE') else None
        return original(*a,**k)
    monkeypatch.setattr(m.synthesis,'propagate',row)
    result=m.run(root,root/'contract.json',inp,out)
    assert not m.io.read(out/'STOP.json')['failed']
    assert {x['status'] for x in result['candidates']['CDF_s1']['rounds']}=={status}
    assert {x['status'] for x in result['candidates']['XDF_s4']['rounds']}=={'POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL'}


def test_full512_frobenius_keeps_out_of_sector_columns_and_uses_no_SVD():
    ideal=np.eye(512,dtype=complex);actual=ideal.copy();actual[0,0]=np.exp(1e-11j)
    # Column0 lies outside H4 Nalpha=Nbeta2; full-space report must retain it.
    with patch.object(np.linalg,'svd',side_effect=AssertionError('NO_FULL_SVD')):
        r=m.synthesis.segment(ideal,actual,ideal,'ARTIFICIAL_full512')
    assert r['input_dimension']==r['output_dimension']==512
    assert r['compiled_reference_frobenius']>9.9e-12
    assert r['operator_bound_pilot']>=abs(actual[0,0]-1)
    assert r['rigorous_operator_bound'] is None and not r['floating_point_rounding_enclosed']
    with pytest.raises(ValueError):m.synthesis.segment(actual*np.nan,actual,ideal,'invalid')


def test_norm_growth_and_16384_repetitions_are_kept():
    k=dict(operator_bound_pilot=1e-11,norm_upper_pilot=1.+2e-11)
    shells={name:dict(operator_bound_pilot=2e-12,norm_upper_pilot=1.+3e-11) for name in m.SHELLS}
    row=m.synthesis.propagate(k,shells,s=4,m=12,saved_pf_error=.01)
    assert row['occurrences']==16384 and row['norm_growth_pilot']>1
    assert row['kernel_repeat_bound_pilot']>16384*k['operator_bound_pilot']
    assert row['combined_complex_error_pilot']==pytest.approx(.01+row['complex_signal_bound_pilot'])
    assert row['CA_achieved'] is None
    bad=dict(k,operator_bound_pilot=None)
    assert m.synthesis.propagate(bad,shells,s=1,m=0,saved_pf_error=.01)['complex_signal_bound_pilot'] is None


def test_common_precision_cache_hit_coverage_and_exact_basis_sharing():
    data,_=artificial_data();c=dict(export='e',saved_inputs={'e':'s'},original_input={'sha256':'i'})
    models=m.prepare_models(data,c,lambda kind,label,fn:fn());builder=m.PrecisionBuilder()
    for arm in m.ARMS:
        base=models[arm].blocks[0]
        models[arm]=replace(models[arm],blocks=tuple(replace(b,rotation=base.rotation) for b in models[arm].blocks))
        for block in models[arm].blocks:m.basis_request(builder,models[arm],block)
    assert len(builder.cache)==2 and len(builder.basis_reports)==18
    assert m.fairness(models,builder)['status']=='PASS'
    assert len(m.native.runs(models['CDF'],m.native.ordered_schedule(9,'1/10'),optimized=True))==1


@pytest.mark.parametrize('kind',['source','input','condition'])
def test_identity_mismatch_preflight_STOPS_before_decode(tmp_path,monkeypatch,kind):
    root,c,inp=contract_fixture(tmp_path,monkeypatch)
    if kind=='source':(root/'code.txt').write_text('CHANGED')
    if kind=='input':inp.write_bytes(b'CHANGED')
    if kind=='condition':
        c['compiler']['seed_transpiler']=18;(root/'contract.json').write_text(json.dumps(c))
    with patch.object(np,'load',side_effect=AssertionError('NO_NUMERIC_DECODE')):
        with pytest.raises(ValueError):m.preflight(root,root/'contract.json',inp)


def test_artificial_future_RPE_cost_connection_uses_same_definitions():
    target=accuracy.AccuracyTarget(.00159,1/8,.1,np.pi/6,np.pi/12,.01/26,.01)
    plans=[accuracy.assess_round(target,accuracy.RoundEvidence(i,1.,.01,0.,1e-6,0.,
        'ARTIFICIAL', 'own_DF_ground_state_oracle',True,'numerically_qualified_pilot')) for i in range(13)]
    energy=accuracy.assess_energy(target,1e-5,plans,state_model='own_DF_ground_state_oracle',alias_branch_verified=True)
    assert energy['same_accuracy'] and energy['numerical_pilot']
    metrics={k:10 for k in accuracy.METRICS}
    k=dict(target_id='ARTIFICIAL',policy='native_kernel_repeat_with_fences',mean=metrics)
    shell=dict(axis='X',policy='separate_native_readout',mean=metrics)
    row=m.connection.compose_serial_cost(k,shell,m=12,substeps=4,target_id='ARTIFICIAL')
    assert row['mean']['rz_count']==16384*10+10
    assert row['depth_semantics']=='scheduled_serial_sum'
    costs=[m.connection.compose_serial_cost(k,dict(shell,axis=axis),m=i,substeps=4,
        target_id='ARTIFICIAL') for i in range(13) for axis in ('X','Y')]
    total=accuracy.aggregate_totals(energy,plans,costs,cost_scope=row['scope'])
    assert total['status']=='CONDITIONAL_TOTAL_AVAILABLE'
    assert accuracy.compare_totals(total,total)['superiority'] is False


def test_real_saved_loader_and_identity_with_artificial_H4_shaped_packet(tmp_path,monkeypatch):
    root,c,inp=contract_fixture(tmp_path,monkeypatch)
    h=np.diag([-.3,-.1,.1,.2]);scalar=np.array(.37)
    np.savez_compressed(inp,one_mo=h,eri_chemist=np.zeros((4,)*4),nuclear_repulsion=scalar,mo_coeff=np.eye(4),
        df_one_body=np.kron(h,np.eye(2)).astype(complex),df_lambdas=np.zeros(10),
        df_g_matrices=np.repeat(np.eye(8,dtype=complex)[None],10,axis=0),forbidden_saved_energy=np.array('DO_NOT_LOAD'))
    monkeypatch.setattr(m.io,'P_H4_SHA256',m.io.sha(inp));c['original_input']['sha256']=m.io.sha(inp)
    arrays=dict(kappa=h,diag_coulomb_mats=np.zeros((8,4,4)),orbital_rotations=np.repeat(np.eye(4)[None],8,axis=0),
        constant=scalar,optimizer_x=np.zeros(128),optimizer_jac=np.zeros(128))
    np.savez_compressed(root/c['export'],**arrays)
    metadata=dict(schema=m.stage.EXPORT_SCHEMA,ffsim_commit=m.stage.FFSIM_COMMIT,
        compatibility_version=m.stage.COMPAT_VERSION,z_representation=False,one_body_role='ffsim_corrected_kappa',
        settings=m.stage.FIT_SETTINGS,optimizer=dict(success=True),input_sha256=m.io.sha(inp),
        source_manifest_sha256=m.io.sha(root/c['fit_SOURCE']),export_sha256=m.io.sha(root/c['export']),
        arrays={k:m.io.array_record(v) for k,v in arrays.items()},actual_rank=8,
        frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)])
    # The existing loader requires its historical filenames; the fixture uses
    # a root-local alias, without modifying any package or old evidence.
    (root/'fit_export.npz').write_bytes((root/c['export']).read_bytes())
    c['export']='fit_export.npz'
    (root/'fit_export.json').write_text(json.dumps(metadata));c['fit_metadata']='fit_export.json'
    for path in c['CDF_operators']:np.savez_compressed(root/path,matrix=np.zeros((256,256),complex))
    np.savez_compressed(root/c['CDF_sector'],matrix=np.zeros((36,36),complex))
    np.savez_compressed(root/c['XDF_generators'],generators=np.zeros((9,36,36),complex))
    psi=np.zeros(36,complex);psi[0]=1.
    for path in c['ground_states'].values():np.savez_compressed(root/path,sector_state=psi)
    for path in c['XDF_kernels'].values():np.savez_compressed(root/path,pf=np.eye(36,dtype=complex))
    rounds=[dict(m=i,pf_operator_error=.001,attenuation=1.,rte_corrected_operator_error=0.) for i in range(13)]
    records={k:{} for k in c['records']}
    records.update(CDF_prepared=dict(df_error_bounds_ha=dict(CDF=1e-5),state_preparation='ORACLE_EXCLUDED'),
        CDF_STOP=dict(failed=False,mandatory_stop=True),CDF_stage1=dict(status='STAGE1_REPRESENTATION_PASS'),
        CDF_provenance=dict(fit_export_sha256=m.io.sha(root/c['export']),input_sha256=m.io.sha(inp)),
        XDF_STOP=dict(failed=False,mandatory_stop=True,scientific_units=93),XDF_SOURCE=dict(execution_commit='ARTIFICIAL'),
        XDF_result=dict(status='XDF8_STAGE1_COMPLETE',representation='PASS',ground=dict(status='PASS'),
            saved_DF=dict(status='PASS',saved_norm_plus_allowance_Ha=1e-5),
            candidates={str(s):dict(summary=dict(status='POTENTIALLY_FEASIBLE'),rounds=rounds) for s in (1,4)}))
    for s in (1,4):records[f'CDF_PF_s{s}']=dict(status='POTENTIALLY_FEASIBLE',candidate=dict(arm='CDF',family='D',s=s,r=0,K=0),rounds=rounds)
    for k,path in c['records'].items():(root/path).write_text(json.dumps(records[k]))
    c['saved_inputs'][c['export']]=m.io.sha(root/c['export'])
    calls=root/'CALLS';calls.mkdir();call=m.integrated.Calls(calls,m.LIMITS,m.MAX_CALLS)
    data=m.load_data(root,c,inp,call);verified=m.identity(data,c)
    assert verified['status']=='PASS' and len(verified['rows'])==4
    assert sum(call.counts.values())==27 and data['XDF_sector'].shape==(9,36,36)
    assert 'forbidden_saved_energy' not in data['packet']
    data['evidence']['CDF_PF_s1']['candidate']['arm']='XDF'
    with pytest.raises(ValueError,match='CDF_PF_TARGET'):m.identity(data,c)
    data['evidence']['CDF_PF_s1']['candidate']['arm']='CDF'
    data['grounds']['CDF']=psi*2
    with pytest.raises(ValueError,match='GROUND_NORMALIZATION'):m.identity(data,c)


@pytest.mark.parametrize('value',[np.eye(4),np.full((4,4),np.nan,dtype=complex),np.eye(3,dtype=complex)])
def test_matrix_schema_nonfinite_refused(tmp_path,value):
    p=tmp_path/'ARTIFICIAL.npz';np.savez_compressed(p,matrix=value)
    with pytest.raises(ValueError,match='SCHEMA'):m.matrix_load(p,'matrix',(4,4))


def test_nonnative_compile_output_is_refused_before_metrics(monkeypatch):
    q=QuantumCircuit(2);q.h(0)
    monkeypatch.setattr(m,'compiled_wrapper',lambda *a:q.copy())
    with pytest.raises(ValueError,match='P0_NATIVE_BASIS'):m.compile_checked(q,m.CompilerProfile())

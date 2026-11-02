"""B3 interactions and saved-evidence dispatch on artificial data only."""
from dataclasses import replace
from fractions import Fraction
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_h4_b3 as m
import test_research_c_c1_h4_native_fair as fixtures


def artificial():
    data, _ = fixtures.artificial_data(); model, reports = m.prepare(data['packet'], 'ARTIFICIAL_INPUT')
    reference = m.fair.reference._ordered_exponentials(data['CDF'], m.native.ordered_schedule(9, '1/10'))
    shells = {}
    for axis in m.fair.SHELLS:
        h, t = m.fair.connection.readout_segments(4, 'Y' if axis == 'Y' else 'X')
        a = m.fair.decode(h if axis == 'head' else t)
        shells[axis] = {k:a for k in ('raw','compiled')}
    return data, model, reports, reference, shells


def mock_evidence(reports):
    metrics = {k:10 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')}
    segment = dict(input_dimension=512, output_dimension=512, global_phase_fit=False, compiled_reference=dict(max_entry=0.))
    candidate = lambda arm:dict(s=1, arm=arm, segment=copy.deepcopy(segment), XY={'compiled_reference':dict(error=0.)}, metrics=dict(metrics))
    previous = dict(baseline_scientific_SOURCE='862a97d4213121749448ca716cde820a44b99772')
    records = dict(result=dict(status='H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE', B3='NOT_RUN', candidates={r:candidate('CDF' if 'CDF' in r else 'XDF') for r in m.strong.ROUTES}),
        SOURCE=dict(execution_commit=m.PREVIOUS_SOURCE), STOP=dict(failed=False, scientific_units=227, retry=0,resume=0,mandatory_stop=True), contract=previous,
        identity=dict(status='PASS',DF_bounds_Ha=dict(CDF=1e-5,XDF=1e-5)),spin=reports,
        B0_result=dict(status='H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE',candidates={a+'_s1':candidate(a) for a in ('CDF','XDF')}),
        B0_SOURCE=dict(execution_commit=previous['baseline_scientific_SOURCE']),B0_STOP=dict(failed=False,scientific_units=305,mandatory_stop=True),
        fairness=dict(status='PASS',compiler=m.fair.profile_dict(),gaussian=m.fair.GAUSSIAN,compiler_phase=m.fair.PHASE,additional_relative_Gaussian_fusion=False))
    return records, previous


def test_passive_import_legacy_routes_and_B3_CLI():
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'src');from trottertracks import c1_h4_b3;assert not any(k in sys.modules for k in ['numpy','scipy','qiskit','ffsim'])"],cwd=ROOT,check=True)
    assert m.strong.ROUTES == ('B1_XDF_spin','B2_CDF_relative','B2_XDF_relative')
    assert m.strong.MAX_CALLS == 227 and m.MAX_CALLS == sum(m.LIMITS.values()) == 65
    r = subprocess.run([sys.executable,'scripts/run_research_c_c1_h4_b3.py','--contract','/NO_READ','--input','/NO_READ','--execute'],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode == 2 and 'Fresh B3 output required' in r.stderr


@pytest.mark.parametrize('time', ['1/10','-1/10'])
def test_both_options_keep_B1_basis_and_B2_semantics_full_space(time):
    data, model, reports, _, _ = artificial()
    original, old_reports = m.strong.prepare_xdf_basis(data['packet']['df_one_body'],data['packet']['df_lambdas'][:8],data['packet']['df_g_matrices'][:8],.37,source_identity='ARTIFICIAL_INPUT:prefix8',input_identity='ARTIFICIAL_INPUT',basis_policy=m.strong.SPIN)
    assert model == original and reports == old_reports
    assert m.model_gate(model,reports,reports)['status'] == 'PASS'
    b1 = m.strong.StrongBuilder(basis_policy=m.strong.SPIN)
    b3 = m.strong.StrongBuilder(basis_policy=m.strong.SPIN,relative_fusion=True)
    old = b1.ordered(model,time,optimized=True); new = b3.ordered(model,time,optimized=True)
    actual,expected = Operator(new).data,Operator(old).data
    assert np.max(abs(actual-expected)) < 1e-13 and np.max(abs(actual[:16,:16]-np.eye(16))) < 1e-13
    assert new.metadata['relative_fusion'] and new.metadata['basis_policy'] == m.strong.SPIN
    assert b1.cache is not b3.cache and b1.relative_cache is not b3.relative_cache
    assert np.max(abs(Operator(m.native.actual_adjoint(new)).data-actual.conj().T)) < 1e-13
    for axis in ('X','Y'):
        a=m.native.wrapper(new,model,time,axis,optimized=True,occupied=(0,2));b=m.native.wrapper(old,model,time,axis,optimized=True,occupied=(0,2))
        assert np.max(abs(m.fair.decode(a)-m.fair.decode(b))) < 1e-13


@pytest.mark.parametrize('defect', ['source','scope','metric','signal','spin','order','nonfinite'])
def test_bad_saved_evidence_and_bad_model_explicitly_refused(defect):
    data, model, reports, _, _ = artificial(); records,previous=mock_evidence(reports)
    if defect=='source': records['SOURCE']['execution_commit']='OTHER'
    if defect=='scope': records['contract']={}
    if defect=='metric': records['result']['candidates']['B1_XDF_spin']['metrics']['rz_count']=float('nan')
    if defect=='signal': records['result']['candidates']['B1_XDF_spin']['XY']['compiled_reference']['error']=.01
    if defect in ('source','scope','metric','signal'):
        with pytest.raises(ValueError):m.saved_identity(records,previous)
    elif defect=='order':
        with pytest.raises(ValueError):m.model_gate(replace(model,blocks=tuple(reversed(model.blocks))),reports,reports)
    else:
        p={k:v.copy() for k,v in data['packet'].items()}
        if defect=='spin':p['df_g_matrices'][0,0,1]=1e-30
        else:p['df_g_matrices'][0,0,0]=np.nan
        with pytest.raises(ValueError):m.prepare(p,'ARTIFICIAL')


def setup_execute(tmp_path):
    data,model,reports,reference,shells=artificial();records,previous=mock_evidence(reports)
    out=tmp_path/'ARTIFICIAL';out.mkdir()
    for n in ('circuits','operators'):(out/n).mkdir()
    call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    builder=m.strong.StrongBuilder(relative_fusion=True,basis_policy=m.strong.SPIN,call=call)
    return data,model,reports,reference,shells,records,previous,out,call,builder


def test_artificial_B3_only_complete_pipeline_all_calls_and_null_qualifications(tmp_path):
    data,model,reports,reference,shells,records,previous,out,call,builder=setup_execute(tmp_path)
    with patch.object(m.strong,'execute',side_effect=AssertionError('NO_OLD_RUNNER')), \
         patch.object(m.fair,'prepare_models',side_effect=AssertionError('NO_CDF')), \
         patch.object(m.io,'load_export',side_effect=AssertionError('NO_CDF_EXPORT')), \
         patch.object(m.fair.synthesis,'propagate',side_effect=AssertionError('NO_ROUNDS')):
        for b in model.blocks:call('gaussian_build',b.block_id,lambda b=b:m.fair.basis_request(builder,model,b))
        r=m.execute(model,data['grounds']['XDF'],reference,shells,builder,m.saved_identity(records,previous),out,call,dict(workers=3))
    assert r['route']==m.ROUTE and r['status']=='H4_B3_S1_NUMERICAL_COMPLETE'
    assert len(r['comparisons'])==3 and r['CA_achieved'] is r['shots'] is r['total_cost'] is r['rigorous_enclosure'] is None
    assert len(list((out/'circuits').glob('*.qpy')))==4 and len(list((out/'operators').glob('*.npz')))==4
    from qiskit import qpy
    with (out/'circuits'/(m.ROUTE+'_raw.qpy')).open('rb') as stream:
        assert qpy.load(stream)[0].metadata['route']==m.ROUTE
    missing={'saved_input_load','saved_evidence_load','saved_identity','native_model','model_gate','reference_load','shell_operator_load','ground_load'}
    for kind,count in call.counts.items():assert count<=m.LIMITS[kind]
    assert set(call.counts)==set(m.LIMITS)-missing
    assert sum(call.counts.values())+sum(m.LIMITS[k] for k in missing)<=65
    assert r['segment']['input_dimension']==32 and not r['segment']['floating_point_rounding_enclosed']


@pytest.mark.parametrize('defect',['phase','direction','compile','reference'])
def test_artificial_failure_STOPS_before_metrics_no_fallback(tmp_path,monkeypatch,defect):
    data,model,reports,reference,shells,records,previous,out,call,builder=setup_execute(tmp_path)
    if defect in ('phase','direction'):
        original=builder.transition
        def bad(model,a,b):
            q=original(model,b,a) if defect=='direction' else original(model,a,b).copy()
            if defect=='phase':q.global_phase+=.03
            return q
        monkeypatch.setattr(builder,'transition',bad)
    elif defect=='compile':
        original=m.fair.compile_checked
        def bad(*a):q=original(*a);q.global_phase+=.03;return q
        monkeypatch.setattr(m.fair,'compile_checked',bad)
    else:reference=reference*np.exp(.03j)
    with pytest.raises(ValueError):m.execute(model,data['grounds']['XDF'],reference,shells,builder,m.saved_identity(records,previous),out,call,dict(workers=3))
    assert call.stop.is_set() and not call.counts['metrics'] and not call.counts['cost_comparison']
    if defect in ('phase','direction','reference'):assert not call.counts['compile']


@pytest.mark.parametrize('fail',[False,True])
def test_full_runner_fresh_partial_evidence_STOP_with_mock_IO(tmp_path,monkeypatch,fail):
    data,model,reports,reference,shells,records,previous,root,_,_=setup_execute(tmp_path)
    c=dict(original_input=dict(sha256='ARTIFICIAL'),source_manifest='source.json',records={k:k+'.json' for k in records},reference='ref.npz',ground='ground.npz',shells={a:{k:a+'_'+k+'.npz' for k in ('raw','compiled')} for a in shells})
    for k,p in c['records'].items():m.io.save(root/p,records[k])
    m.io.save(root/'source.json',dict(files={}))
    monkeypatch.setattr(m,'preflight',lambda *a:(c,previous))
    monkeypatch.setattr(m.io,'load_input',lambda *a,**k:data['packet'])
    def load(p,member,shape):
        name=Path(p).name
        if name=='ref.npz':return reference
        if name=='ground.npz':return data['grounds']['XDF']
        a,k=name[:-4].rsplit('_',1);return shells[a][k]
    monkeypatch.setattr(m.fair,'matrix_load',load)
    original=m.subprocess.check_output
    monkeypatch.setattr(m.subprocess,'check_output',lambda cmd,*a,**k:'ARTIFICIAL_SOURCE\n' if cmd[:2]==['git','rev-parse'] else original(cmd,*a,**k))
    if fail:monkeypatch.setattr(m,'model_gate',lambda *a:(_ for _ in ()).throw(ValueError('ARTIFICIAL_GATE_FAIL')))
    out=root/m.FAMILY/'2026-10-26/run01'
    if fail:
        with pytest.raises(ValueError,match='GATE_FAIL'):m.run(root,root/'c',root/'input',out)
    else:assert m.run(root,root/'c',root/'input',out)['status']=='H4_B3_S1_NUMERICAL_COMPLETE'
    stop=m.io.read(out/'STOP.json');assert stop['failed']==fail and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert (out/'SOURCE.json').exists() and (out/'failure.json').exists()==fail
    with pytest.raises(ValueError,match='FRESH'):m.run(root,root/'c',root/'input',out)


@pytest.mark.parametrize('defect',['none','route','source','reference'])
def test_byte_preflight_never_decodes_saved_science(tmp_path,monkeypatch,defect):
    root=tmp_path/'ARTIFICIAL';root.mkdir()
    names=['prev.json','ref.npz','shell.npz','ground.npz']+['old_'+k+'.json' for k in m.RECORDS]
    for p in names:(root/p).write_bytes(b'ARTIFICIAL_NONNUMERIC_BYTES')
    (root/'module.py').write_text('ARTIFICIAL_SOURCE')
    m.io.save(root/'manifest.json',dict(files={'module.py':m.io.sha(root/'module.py')}))
    shells={a:{k:'shell.npz' for k in ('raw','compiled')} for a in m.fair.SHELLS}
    previous=dict(original_input=dict(sha256='ARTIFICIAL'),references=dict(XDF='ref.npz'),shells=shells)
    c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT_DRAFT',route=m.ROUTE,comparators=list(m.COMPARATORS),
        conditions=copy.deepcopy(m.CONDITIONS),allowances=m.strong.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        previous_scientific_SOURCE=m.PREVIOUS_SOURCE,previous_publication=m.PREVIOUS_PUBLICATION,
        compiler=m.fair.profile_dict(),gaussian=m.fair.GAUSSIAN,compiler_phase=m.fair.PHASE,
        fresh_output=m.FAMILY+'/<execution-date>/run01',source_manifest='manifest.json',source_manifest_sha256=m.io.sha(root/'manifest.json'),
        saved_inputs={p:m.io.sha(root/p) for p in names},previous_contract='prev.json',reference='ref.npz',ground='ground.npz',shells=shells,
        records={k:'old_'+k+'.json' for k in m.RECORDS},original_input=previous['original_input'],**{k:0 for k in m.FORBIDDEN})
    if defect=='route':c['route']='B2_CDF_relative'
    if defect=='source':(root/'module.py').write_text('CHANGED')
    if defect=='reference':c['reference']='ground.npz'
    m.io.save(root/'contract.json',c)
    monkeypatch.setattr(m.strong,'preflight',lambda *a:(previous,dict(ground_states=dict(XDF='ground.npz'))))
    with patch.object(np,'load',side_effect=AssertionError('NO_NUMERIC_DECODE')):
        if defect=='none':assert m.preflight(root,root/'contract.json',root/'packet')[0]==c
        else:
            with pytest.raises(ValueError):m.preflight(root,root/'contract.json',root/'packet')

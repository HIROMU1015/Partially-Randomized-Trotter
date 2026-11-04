"""Invented gate records/processes only; never load molecular inputs or QPY."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest
from trottertracks import c_stage_p_completion as m
from trottertracks import c_pr_aware_df_fixed_certificate as fixed


@pytest.mark.parametrize('phase',[0.,.125])
def test_progress_arithmetic_matches_existing_engine(tmp_path,phase):
    a=fixed.identity(2);a['phase_hex']=phase.hex()
    a['gates']=[dict(name='rz',qubits=[0],params_hex=[(.3).hex()]),dict(name='cx',qubits=[0,1],params_hex=[])]
    b=deepcopy(a);b['gates'][0]['params_hex']=[(.3+2**-30).hex()]
    expected=fixed.compare(a,b,parallel=False)
    actual=m.compare_progress(a,b,tmp_path/'columns',parallel=False)
    for key in ('columns','upper_exact','lower_exact','precision_bits','dimension','phase_fit'):
        assert actual[key]==expected[key]
    rows=[json.loads(s) for s in (tmp_path/'columns/columns.jsonl').read_text().splitlines()]
    assert rows==expected['columns']
    progress=json.loads((tmp_path/'columns/progress.json').read_text())
    assert progress['completed_columns']==progress['total_columns']==4 and not progress['restart_allowed']
    with pytest.raises(FileExistsError):m.compare_progress(a,b,tmp_path/'columns',parallel=False)


def test_full512_parallel_progress_matches_serial_invented_phase(tmp_path):
    a=fixed.identity(9);b=deepcopy(a)
    a['gates']=[dict(name='rz',qubits=[8],params_hex=[(.125).hex()])]
    b['gates']=[dict(name='rz',qubits=[8],params_hex=[(.125+2**-35).hex()])]
    expected=fixed.compare(a,b,parallel=False)
    actual=m.compare_progress(a,b,tmp_path/'columns',parallel=True)
    assert actual['columns']==expected['columns']
    assert actual['upper_exact']==expected['upper_exact']
    assert actual['lower_exact']==expected['lower_exact']
    assert len((tmp_path/'columns/columns.jsonl').read_text().splitlines())==512


def test_progress_partial_failure_is_saved_without_resume(tmp_path,monkeypatch):
    original=fixed._pair_column
    def fail(i):
        if i==2:raise ValueError('invented column FAIL')
        return original(i)
    monkeypatch.setattr(fixed,'_pair_column',fail)
    with pytest.raises(ValueError,match='invented column FAIL'):
        m.compare_progress(fixed.identity(2),fixed.identity(2),tmp_path/'columns',parallel=False)
    assert len((tmp_path/'columns/columns.jsonl').read_text().splitlines())==2
    assert json.loads((tmp_path/'columns/progress.json').read_text())['completed_columns']==2


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    c=dict(fresh_output='run',source_manifest='SOURCE.json')
    (tmp_path/'contract.json').write_text(json.dumps(c));(tmp_path/'SOURCE.json').write_text('{}')
    monkeypatch.setattr(m,'preflight',lambda *args,**kwargs:dict(status='invented'))
    return tmp_path,tmp_path/'contract.json'


@pytest.mark.parametrize('fault',['none','exception','SIGTERM'])
def test_fresh_lifecycle_STOP_and_signals(lifecycle,fault):
    root,path=lifecycle
    def execute(root,c,out,call):
        def value():
            if fault=='exception':raise ValueError('invented failure')
            if fault=='SIGTERM':signal.raise_signal(signal.SIGTERM)
            return dict(status='invented success')
        return call('summary','invented',value)
    if fault=='none':assert m.run(root,path,execute_science=True,executor=execute)['status']=='invented success'
    else:
        with pytest.raises((ValueError,InterruptedError)):m.run(root,path,execute_science=True,executor=execute)
    stop=json.loads((root/'run/STOP.json').read_text())
    assert stop['scientific_units']==1 and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert bool(stop['failed'])==(fault!='none')
    if fault=='SIGTERM':assert json.loads((root/'run/termination_signal.json').read_text())['signal']==signal.SIGTERM


def test_missing_permission_stops_before_any_work(tmp_path):
    with pytest.raises(PermissionError):m.run(tmp_path,tmp_path/'absent.json')
    assert not list(tmp_path.iterdir())


@pytest.fixture
def launcher():
    path=Path(__file__).resolve().parents[2]/m.ENTRY
    spec=importlib.util.spec_from_file_location('invented_p_launcher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('code',[0,7,-signal.SIGTERM])
def test_supervisor_records_exit_and_signal_without_retry(tmp_path,launcher,code):
    script='import sys;sys.exit(%d)'%code if code>=0 else 'import os,signal;os.kill(os.getpid(),signal.SIGTERM)'
    result=launcher.supervise([sys.executable,'-c',script],tmp_path/'launch',poll_seconds=.02)
    assert result==code
    saved=json.loads((tmp_path/'launch/exit_status.json').read_text())
    assert saved['returncode']==code and saved['terminating_signal']==(-code if code<0 else None)
    assert saved['retry']==saved['resume']==0 and saved['mandatory_stop']
    with pytest.raises(FileExistsError):launcher.supervise([sys.executable,'-c','pass'],tmp_path/'launch')


def invented_evidence():
    C=fixed.identity(9);digest=m.perword.literal._digest(C)
    columns=[dict(column=i) for i in range(512)]
    base=dict(status='VALIDATED_FIXED_CIRCUIT_PAIR',dimension=512,columns_evaluated=512,
              precision_bits=192,phase_fit=False,columns=columns,upper_exact='1/1000')
    cr=dict(base,left_id=digest,right_id=digest)
    rd=dict(base,status='VALIDATED_RAW_LITERAL_OPERATOR',raw_id=digest,target_id='toy',word_id='toy-word')
    binding=dict(target_id='toy',word_id='toy-word',word=[['a',(.125).hex()]]*32)
    b=dict(status='VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR',binding=binding,
           compiled_id=digest,raw_id=digest,compiled_raw=cr,raw_literal=rd,upper_exact='1/500',
           all_word_uniform=False,Head_bridge_to_B2_included=False,phase_fit=False)
    done=[('records','compiled'),('adjoint_record','compiled'),('compiled_raw_certificate','full512 columns'),
          ('raw_literal_certificate','full512 columns;no cap2'),('perword_triangle','two legs once;Head-to-B2 not included')]
    done += [('invented','unused'+str(i)) for i in range(24)]
    journal=[]
    for kind,label in done:
        journal.extend([dict(event='attempt',kind=kind,label=label),dict(event='COMPLETE',kind=kind,label=label)])
    journal.append(dict(event='attempt',kind='adjoint_certificate',label='compiled'))
    e={'compiled_raw.json':cr,'raw_literal.json':rd,'per_word.json':b,
       'raw_actual_inverse.json':dict(base,literal_inverse_claimed=False),
       'literal_reference.json':dict(binding=binding),'raw_record.json':C,
       'population':dict(target_id='toy',word_length=32,D_norm_upper_exact='1',d_range_upper_exact='2',
                         event_angle_bits={'a':(.125).hex()}),'journal':journal}
    return e,(C,deepcopy(C)),dict(word=['a']*32)


@pytest.mark.parametrize('fault',['none','phase','column','sum','word','journal','population'])
def test_saved_evidence_identity_and_mismatch_STOP(fault):
    e,records,c=invented_evidence()
    if fault=='phase':records[0]['phase_hex']=(.125).hex()
    if fault=='column':e['raw_literal.json']['columns']=e['raw_literal.json']['columns'][:-1]
    if fault=='sum':e['per_word.json']['upper_exact']='3/1000'
    if fault=='word':c['word']=['b']*32
    if fault=='journal':e['journal'][-1]['event']='COMPLETE'
    if fault=='population':e['population']['d_range_upper_exact']='3'
    if fault=='none':assert m.identity(e,records,c)['finite_word_only']
    else:
        with pytest.raises(ValueError):m.identity(e,records,c)


def test_completion_pipeline_exact15_calls_no_build_compile(tmp_path,monkeypatch):
    from qiskit import QuantumCircuit,qpy
    from trottertracks.c1_stage3_integrated import Calls
    e,records,c=invented_evidence();c['parent_files']={}
    for name,r in zip(('compiled_record.json','compiled_adjoint_record.json'),records):
        (tmp_path/name).write_text(json.dumps(r));c['parent_files'][name]=name
    c['shells']={}
    for label in ('head','X','Y'):
        q=QuantumCircuit(9)
        if label=='Y':q.sdg(8)
        q.h(8);c['shells'][label]={}
        for kind in ('raw','compiled'):
            path=tmp_path/(label+'_'+kind+'.qpy')
            with path.open('wb') as f:qpy.dump(q,f)
            c['shells'][label][kind]=path.name
    monkeypatch.setattr(m,'evidence',lambda *args:e)
    def compare(left,right,out):
        return dict(status='INVENTED_ROUTING_ONLY',upper_exact='1/1000000')
    monkeypatch.setattr(m,'compare_progress',compare)
    out=tmp_path/'fresh';out.mkdir();calls=Calls(out,m.LIMITS,m.MAXIMUM)
    value=m.execute(tmp_path,c,out,calls)
    assert sum(calls.counts.values())==m.MAXIMUM==15
    assert calls.counts['compiled_adjoint_certificate']==1 and calls.counts['shell_load']==6
    assert calls.counts['shell_certificate']==3 and not any(k in calls.counts for k in ('compile','kernel_build','sampling'))
    assert value['missing_outputs_obtained'] and not value['original_41_call_run_complete']
    assert value['finite_signal']['occurrences']==1 and not value['finite_signal']['all_law_synthesis_qualified']
    assert F(value['compiled_actual_inverse']['to_literal_adjoint_upper_exact'])==F('1/1000000')+F('1/500')


def test_real_contract_passive_cold_no_numeric_imports():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_stage_p_completion as m
r=m.preflight(Path.cwd(),m.CONTRACT)
assert r['scientific_calls']==r['arrays_decoded']==0 and r['maximum_logical_units']==15
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('fault',['input','source','typed','model','word','parent_role','retry','precision'])
def test_real_metadata_preflight_mismatch_STOP_without_decoding(tmp_path,fault):
    root=Path(__file__).resolve().parents[2];c=m.io.read(root/m.CONTRACT)
    if fault=='input':c['saved_inputs'][next(iter(c['saved_inputs']))]='0'*64
    if fault=='source':c['source_manifest_sha256']='0'*64
    if fault=='typed':c['maximum_logical_units']=15.0
    if fault=='model':c['identity']['model']['independent_events']=1
    if fault=='word':c['word']=['CDF_proposed:F6:0']*32
    if fault=='parent_role':c['parent_files']['compiled_record.json']=c['parent_files']['compiled_adjoint_record.json']
    if fault=='retry':c['retry']=1
    if fault=='precision':c['precision_bits']=128
    path=tmp_path/'contract.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError):m.preflight(root,path)

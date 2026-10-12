"""Synthetic metadata, lossless I/O and frozen-code controls only."""
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_saved_recovery import storage,launch
from trottertracks.algorithm_codesign.df_correction_et_saved_recovery.accounting import original_block
from trottertracks.algorithm_codesign.df_correction_comparison.native import t_completion,local_t_separation
from trottertracks.algorithm_codesign.df_correction_comparison.angles import Angle

SOURCE=ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v7.py'


def test_large_unknown_dictionary_is_lossless_and_review_bounded(tmp_path):
    value={'rows':[{'A':{'unknown_multiplicities':{'raw-rad:'+str(i)+'/123456789:scale:1':i%4+1 for i in range(25000)}}}]}
    raw=(json.dumps(value,default=str,indent=2)+'\n').encode()
    assert len(raw)>512*1024
    path=tmp_path/'T_accounting.json'
    storage.dump_json(path,value)
    assert storage.load_json(path)==value
    assert b''.join(storage.iter_verified_bytes(path))==raw
    assert storage.artifact_identity(path)['sha256']==hashlib.sha256(raw).hexdigest()
    assert all(p.stat().st_size<=512*1024 for p in tmp_path.rglob('*') if p.is_file())


@pytest.mark.parametrize('operation',['synthesize','transpile','Operator','native_fixture'])
def test_source_block_rejects_scientific_calls(operation):
    source=SOURCE.read_text().replace('accounting=[]','accounting=['+operation+'(None)]')
    with pytest.raises(PermissionError):original_block(source)


def test_original_block_runs_only_synthetic_metadata_and_preserves_unknown_phase(monkeypatch):
    from trottertracks.algorithm_codesign.df_correction_comparison import angles
    def reject(*a,**k):raise AssertionError('No synthesis in metadata reconstruction')
    monkeypatch.setattr(angles,'synthesize',reject)
    angles_list=[Angle('atan',F(1,128),F(s)) for s in (-1,1,-2,2)]
    seq=[]
    for i,a in enumerate(angles_list):
        text='T'*(i+1);seq.append({'angle':a.as_dict(),'sequence':text,'sequence_sha256':hashlib.sha256(text.encode()).hexdigest(),'T_count':i+1,'Tdagger_count':0,'W_count':0,'error_pass':True,'strict_operator_error_upper':'0'})
    jobs=[{'label':'T-'+str(i),'output':'synthetic-'+str(i)} for i in range(4)]
    native={r:{'ell_computational':'1','B_local_numeric_uniform_upper':'0','all_B_event_costs':[{'controlled':c,'sign':1,'executed_conditional_probability':'1','total_RZ':1 if not c else 2} for c in (False,True)]} for r in (3,6)}
    circuits={(r,c):{'occurrences':[],'phase_supported':False,'numeric_operator_error_upper':'0','total_RZ':0} for r in (3,6) for c in (False,True)}
    captured=[]
    context={'F':F,'Angle':Angle,'jobs':jobs,'native':native,'circuits':circuits,'binding':{},'inventory':{'unique_keys':[a.as_dict() for a in angles_list],'native_result_bindings':[]},'load_json':lambda p:seq[int(p.split('-')[-1])],
             'artifact_identity':lambda p:{'logical_path':p},'verify_synthesis_result':lambda *a:None,
             'verify_inventory':lambda *a,**k:None,'out':Path('/tmp/synthetic'),
             'marker_hash':'synthetic','provenance':{},'contract':{'T':{'gate_precision':'1/40000000000'},'caps':{'synthesis':{'sequence_characters':20000}},'E':{'local_numeric_allowance':'1/5000000','local_synthesis_allowance':'1/1250000'}},
             't_completion':t_completion,'local_t_separation':local_t_separation,'dump_json':lambda p,v:captured.append(v)}
    exec(compile(original_block(SOURCE.read_text()),'original_metadata_block','exec'),context)
    assert len(captured)==1 and captured[0]['CD_PR_total_winner'] is None
    assert len(captured[0]['rows'])==4
    for row in captured[0]['rows']:
        assert row['A']['actual_total_T'] is None
        assert row['A']['fixed_plan_completion_T_lower'] is None
        assert row['B']['actual_total_T']==(3 if row['controlled'] else 3)
        assert row['PR_total_winner'] is None


def test_pending_recovery_launch_rejects_before_marker():
    assert not (ROOT/launch.RESULT).exists()
    with pytest.raises(PermissionError):launch.verify_local(ROOT)
    assert not (ROOT/launch.RESULT).exists()
